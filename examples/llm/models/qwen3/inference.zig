const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const Phase = common.Phase;
const model = @import("model.zig");

const log = std.log.scoped(.qwen3);
pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, shardings: common.Shardings) CompilationParameters {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        return .{
            .prefill_tokens = .init(.{ .b = 1, .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(config, 1, seqlen, dtype, shardings.model),
            .rng = .init(),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const Args = struct {
    io: std.Io,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill: KernelExe,
    decode: KernelExe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        qwen_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const prefill = try compileKernel(allocator, io, platform, qwen_model, parameters, @intCast(parameters.prefill_tokens.dim(.s)), .prefill, progress);
        errdefer prefill.deinit();
        const decode = try compileKernel(allocator, io, platform, qwen_model, parameters, @intCast(parameters.decode_tokens.dim(.s)), .decode, progress);
        return .{
            .loaded_model = loaded_model,
            .prefill = prefill,
            .decode = decode,
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

pub const Inference = CompiledModel;

const LayerExe = zml.FnExe(model.TransformerLayer.forwardStep);
const LayerRunner = LayerExe.Runner(.{.layer});

pub const KernelExe = struct {
    embed: zml.FnExe(model.EmbedTokens.forward),
    layer: LayerExe,
    sample: zml.FnExe(model.Sampler.sampleTokens),

    pub fn deinit(self: *const KernelExe) void {
        self.embed.deinit();
        self.layer.deinit();
        self.sample.deinit();
    }
};

pub const KernelRunner = struct {
    embed: zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}),
    layers: []LayerRunner,
    sample: zml.FnExe(model.Sampler.sampleTokens).Runner(.{.sampler}),

    pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const model.Buffers) !KernelRunner {
        var embed = try zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}).init(&exe.embed, allocator, .{
            .embedding = .{ .embed_tokens = buffers.text_model.embed_tokens },
        });
        errdefer embed.deinit(allocator);

        const layers = try allocator.alloc(LayerRunner, buffers.text_model.layers.len);
        errdefer allocator.free(layers);
        var initialized_layers: usize = 0;
        errdefer for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);
        for (layers, buffers.text_model.layers) |*layer, layer_buffers| {
            layer.* = try LayerRunner.init(&exe.layer, allocator, .{ .layer = layer_buffers });
            initialized_layers += 1;
        }

        var sample = try zml.FnExe(model.Sampler.sampleTokens).Runner(.{.sampler}).init(&exe.sample, allocator, .{
            .sampler = .{
                .norm = buffers.text_model.norm,
                .lm_head = buffers.lm_head,
            },
        });
        errdefer sample.deinit(allocator);

        return .{ .embed = embed, .layers = layers, .sample = sample };
    }

    pub fn deinit(self: *KernelRunner, allocator: std.mem.Allocator) void {
        self.embed.deinit(allocator);
        for (self.layers) |*layer| layer.deinit(allocator);
        allocator.free(self.layers);
        self.sample.deinit(allocator);
    }
};

pub fn run(runner: *KernelRunner, args: Args, layer_index_buffers: []const zml.Buffer) void {
    var hidden_buffer: zml.Buffer = undefined;
    runner.embed.run(args.io, .{
        .inputs = .{
            .tokens = args.tokens_buf.*,
        },
        .outputs = .{ .hidden = &hidden_buffer },
    }, .{});
    defer hidden_buffer.deinit();

    for (runner.layers, layer_index_buffers) |*layer, layer_index_buffer| {
        var layer_cache: zml.Bufferized(model.KvCache) = .{
            .k = args.kv_cache_buffers.k,
            .v = args.kv_cache_buffers.v,
            .layer_index = layer_index_buffer,
        };
        layer.run(args.io, .{
            .inputs = .{
                .hidden = hidden_buffer,
                .token_index = args.token_index_buf.*,
                .cache = layer_cache,
            },
            .outputs = .{ .hidden = &hidden_buffer, .cache = &layer_cache },
        }, .{});
        args.kv_cache_buffers.k = layer_cache.k;
        args.kv_cache_buffers.v = layer_cache.v;
    }

    runner.sample.run(args.io, .{
        .inputs = .{
            .hidden = hidden_buffer,
            .rng = args.rng_buffers.*,
            .token_index = args.token_index_buf.*,
        },
        .outputs = .{
            .tokens = args.tokens_buf,
            .rng = args.rng_buffers,
            .token_index = args.token_index_buf,
        },
    }, .{});
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    parameters: CompilationOptions,
    seqlen: usize,
    phase: Phase,
    progress: *std.Progress.Node,
) !KernelExe {
    const embed = try compileEmbed(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
    errdefer embed.deinit();
    const layer = try compileLayer(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
    errdefer layer.deinit();
    const sample = try compileSample(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
    errdefer sample.deinit();
    return .{ .embed = embed, .layer = layer, .sample = sample };
}

fn compileEmbed(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationOptions, seqlen: usize, phase: Phase, progress: *std.Progress.Node) !zml.FnExe(model.EmbedTokens.forward) {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("embed tokens"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "embed tokens", io, from);
    return zml.FnExe(model.EmbedTokens.forward).compile(allocator, io, platform, .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName("qwen3", "embed_tokens") }, .{.{
        .embedding = .{ .embed_tokens = mdl.text_model.embed_tokens },
        .tokens = zml.Tensor.init(.{ .b = 1, .s = seqlen }, .u32),
    }});
}

fn compileLayer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationOptions, seqlen: usize, phase: Phase, progress: *std.Progress.Node) !LayerExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("transformer layer"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "transformer layer", io, from);
    return LayerExe.compile(allocator, io, platform, .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName("qwen3", "layer") }, .{.{
        .layer = mdl.text_model.layers[0],
        .hidden = hiddenTensor(mdl, seqlen),
        .token_index = parameters.token_index,
        .cache = .{
            .k = parameters.kv_cache.k,
            .v = parameters.kv_cache.v,
            .layer_index = zml.Tensor.init(.{}, .u32),
        },
    }});
}

fn compileSample(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationOptions, seqlen: usize, phase: Phase, progress: *std.Progress.Node) !zml.FnExe(model.Sampler.sampleTokens) {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("sampler"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "sampler", io, from);
    return zml.FnExe(model.Sampler.sampleTokens).compile(allocator, io, platform, .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName("qwen3", "sampler") }, .{.{
        .sampler = mdl.sampler(),
        .hidden = hiddenTensor(mdl, seqlen),
        .rng = parameters.rng,
        .token_index = parameters.token_index,
    }});
}

fn hiddenTensor(mdl: model.Model, seqlen: usize) zml.Tensor {
    return .fromShape(zml.Shape.init(
        .{ .b = 1, .s = seqlen, .d = mdl.config.hidden_size },
        mdl.text_model.embed_tokens.weight.dtype(),
    ).withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated }));
}
