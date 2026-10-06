const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.qwen3);

pub const RopeParameters = struct {
    rope_theta: f32 = 1_000_000.0,
};

pub const Config = struct {
    num_hidden_layers: i64,
    hidden_size: i64,
    max_position_embeddings: i64,
    num_attention_heads: i64,
    rms_norm_eps: f32 = 1e-6,
    num_key_value_heads: ?i64 = null,
    head_dim: ?i64 = null,
    rope_parameters: ?RopeParameters = null,
    rope_theta: ?f32 = null,
    use_sliding_window: bool = false,

    pub fn headDim(self: Config) i64 {
        return self.head_dim orelse @divExact(self.hidden_size, self.num_attention_heads);
    }

    pub fn numKvHeads(self: Config) i64 {
        return self.num_key_value_heads orelse self.num_attention_heads;
    }

    pub fn ropeTheta(self: Config) f32 {
        if (self.rope_parameters) |p| return p.rope_theta;
        return self.rope_theta orelse 1_000_000.0;
    }
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        if (parsed_config.value.use_sliding_window) return error.SlidingWindowNotSupported;

        const options: Model.GenOptions = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.max_position_embeddings,
        };

        var inner: Model = try .init(allocator, store, parsed_config.value, options);
        inner.enable_thinking = generation.enable_thinking;

        return .{
            .inner = inner,
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);

        var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
        errdefer self.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        const all_shardings = shardings.all();
        try loader.load(io, Model, &self.inner, &buffers, store, &all_shardings, .{ .progress = progress });
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        TextModel.unloadBuffers(&buffers.text_model, allocator);
        zml.nn.Linear.unloadBuffers(&buffers.lm_head);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), shardings);
        return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    pub const GenOptions = struct { sampling_strategy: zml.nn.SamplingStrategy = .{}, max_seq_len: i64 };

    pub const SpecialTokens = struct {
        im_start_token_id: u32,
        im_end_token_id: u32,
        end_of_text_token_id: u32,
    };

    text_model: TextModel,
    lm_head: zml.nn.Linear,

    config: Config,
    gen_options: GenOptions,
    enable_thinking: bool = true,
    special_tokens: SpecialTokens = .{
        .im_start_token_id = 151644,
        .im_end_token_id = 151645,
        .end_of_text_token_id = 151643,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Model {
        // Qwen/Qwen3-0.6B checkpoints tie lm_head to the input embedding (no `lm_head.weight` in the files).
        // Qwen3-8B has `tie_word_embeddings: false` and provides the `lm_head` weight.
        const lm_head_prefix = b: {
            if (store.hasKey("lm_head.weight")) break :b "lm_head";
            break :b "model.embed_tokens";
        };
        return .{
            .text_model = try .init(allocator, store.withPrefix("model"), config),
            .lm_head = .init(
                store.withPrefix(lm_head_prefix).createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .config = config,
            .gen_options = gen_options,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.text_model.deinit(allocator);
    }

    pub fn forward(
        self: Model,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const hidden, const updated_kv_cache = self.text_model.forward(tokens, token_index, kv_cache);
        const result = Sampler.sampleTokens(.{
            .sampler = self.sampler(),
            .hidden = hidden,
            .rng = rng,
            .token_index = token_index,
        });
        return .{ result.tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, result.rng };
    }

    pub fn sampler(self: Model) Sampler {
        return .{
            .norm = self.text_model.norm,
            .lm_head = self.lm_head,
            .gen_options = self.gen_options,
        };
    }
};

pub const Sampler = struct {
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    gen_options: Model.GenOptions,

    pub const Input = struct {
        sampler: Sampler,
        hidden: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: zml.Tensor,
    };

    pub const Output = struct {
        tokens: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: zml.Tensor,
    };

    pub fn sampleTokens(input: Input) Output {
        const self = input.sampler;
        const x = self.norm.forward(input.hidden);
        const logits = self.lm_head.forward(x.withPartialTags(.{.d}), x.dtype()).rename(.{ .dout = .voc });
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, input.rng);
        return .{
            .tokens = next_tokens.convert(.u32),
            .rng = new_rng,
            .token_index = input.token_index.addConstant(1),
        };
    }
};

pub const EmbedTokens = struct {
    embed_tokens: zml.nn.TokenEmbedding,

    pub const Input = struct {
        embedding: EmbedTokens,
        tokens: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn forward(input: Input) Output {
        const tokens = input.tokens.withPartialTags(.{.s});
        return .{ .hidden = input.embedding.embed_tokens.forward(tokens)
            .withPartialTags(.{.d})
            .withPartitioning(.{ .d = .replicated }) };
    }
};

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
    ) !TextModel {
        const layers = try allocator.alloc(TransformerLayer, @intCast(config.num_hidden_layers));
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .layers = layers,
            .norm = RmsNorm.init(store.withPrefix("norm"), config.rms_norm_eps),
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(
        self: TextModel,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        var hidden_states = EmbedTokens.forward(.{
            .embedding = .{ .embed_tokens = self.embed_tokens },
            .tokens = tokens,
        }).hidden;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forward(hidden_states, token_index, updated_kv_cache.atLayer(i));
        }

        return .{ hidden_states, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    mlp: Mlp,
    post_attention_layernorm: RmsNorm,

    pub const StepInput = struct {
        layer: TransformerLayer,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        cache: KvCache,
    };

    pub const StepOutput = struct {
        hidden: zml.Tensor,
        cache: KvCache,
    };

    pub fn init(store: zml.io.TensorStore.View, config: Config) TransformerLayer {
        return .{
            .input_layernorm = RmsNorm.init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = .init(store.withPrefix("self_attn"), config),
            .mlp = .init(store.withPrefix("mlp")),
            .post_attention_layernorm = RmsNorm.init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.self_attn);
        Mlp.unloadBuffers(&self.mlp);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const normalized_x0 = self.input_layernorm.forward(x0_replicated);

        const attention_output, const updated_kv_cache = self.self_attn.forward(normalized_x0, token_index, kv_cache);

        const x1 = attention_output.add(x0_replicated).withPartitioning(.{ .d = .replicated });
        const normalized_hidden = self.post_attention_layernorm.forward(x1);
        const mlp_output = self.mlp.forward(normalized_hidden).withPartitioning(.{ .d = .replicated });

        return .{ mlp_output.add(x1).withPartitioning(.{ .d = .replicated }).reuseBuffer(x0), updated_kv_cache };
    }

    pub fn forwardStep(input: StepInput) StepOutput {
        const hidden, const cache = input.layer.forward(input.hidden, input.token_index, input.cache);
        return .{ .hidden = hidden, .cache = cache };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(
                store.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                store.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }),
                .d,
            ),
            .gate_proj = .init(
                store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                store.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }),
                .d,
            ),
            .down_proj = .init(
                store.withPrefix("down_proj").createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .model }),
                store.withPrefix("down_proj").maybeCreateTensor("bias", .{.d}, .{ .d = .replicated }),
                .dout,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        zml.Buffer.deinitAll(Mlp, self);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const up_projed = self.up_proj.forward(x, x.dtype());
        const gate = self.gate_proj.forward(x, x.dtype());
        const hidden = gate.silu().mul(up_projed);
        return self.down_proj.forward(hidden, hidden.dtype());
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_embed: RotaryEmbedding,

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype, bias_partitions: anytype) zml.nn.Linear {
        return .init(
            store.createTensor("weight", .{ .dout, .d }, partitions),
            store.maybeCreateTensor("bias", .{.dout}, bias_partitions),
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, config: Config) SelfAttn {
        const head_dim = config.headDim();
        return .{
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .model }, .{ .dout = .replicated }),
            .q_norm = RmsNorm.init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = RmsNorm.init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = config.num_attention_heads,
            .num_kv_heads = config.numKvHeads(),
            .head_dim = head_dim,
            .rotary_embed = .init(head_dim, config.ropeTheta()),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        zml.Buffer.deinitAll(SelfAttn, self);
    }

    fn projectQ(self: SelfAttn, x: zml.Tensor) zml.Tensor {
        return self.q_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim });
    }

    fn projectKV(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const k = self.k_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        const v = self.v_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        return .{ k, v };
    }

    fn partitionProjectedKv(kv: zml.Tensor, kv_head_sharding: zml.Sharding.DimSharding) zml.Tensor {
        return switch (kv_head_sharding) {
            .sharded => |heads| blk: {
                const sharded_kv = if (heads.factor != 1) kv.stutter1d(kv.axis(.h), heads.factor) else kv;
                break :blk sharded_kv.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
            },
            .replicated => kv.withPartitioning(.{ .s = .replicated, .h = .replicated, .hd = .replicated }),
        };
    }

    fn partitionCachedKv(tensor: zml.Tensor, kv_head_sharding: zml.Sharding.DimSharding) zml.Tensor {
        var cache_tensor = tensor.rename(.{ .s = .k });
        return switch (kv_head_sharding) {
            .sharded => |heads| blk: {
                if (heads.factor != 1) {
                    cache_tensor = cache_tensor.stutter1d(cache_tensor.axis(.h), heads.factor);
                }
                break :blk cache_tensor.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
            },
            .replicated => cache_tensor.withPartitioning(.{ .k = .replicated, .h = .replicated, .hd = .replicated }),
        };
    }

    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.projectQ(x_qkv);
        var k, var v = self.projectKV(x_qkv);
        const kv_head_sharding = zml.Compiler.current().partitioning.shardableDim(
            k.shape().withPartitioning(.{ .h = .model }),
            .h,
            q.dim(.h),
        ) catch unreachable;

        k = partitionProjectedKv(k, kv_head_sharding);
        v = partitionProjectedKv(v, kv_head_sharding);

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const dtype = q.dtype();
        const pos_shape = zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64);
        const position_ids = zml.Tensor.arange(.{ .end = x.dim(.s) }, .i64)
            .withTags(.{.s}).insertAxes(.s, .{.b}).broad(pos_shape)
            .add(token_index.convert(.i64).broad(pos_shape));

        const cos, const sin = self.rotary_embed.getCosAndSin(position_ids, dtype);
        q = self.rotary_embed.applyRope(q, cos, sin);
        k = self.rotary_embed.applyRope(k, cos, sin);
        k = partitionProjectedKv(k, kv_head_sharding);
        v = partitionProjectedKv(v, kv_head_sharding);

        const new_kv_cache = kv_cache.update(k, v, token_index.convert(.u32));
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        q = q.rename(.{ .s = .q });
        k = partitionCachedKv(k, kv_head_sharding);
        v = partitionCachedKv(v, kv_head_sharding);

        const attn_output = zml.attention.attention(
            q,
            k,
            v,
            token_index,
            zml.attention.Metadata.init(.fromBackend(.vanilla, x.dim(.s), self.num_heads)),
            zml.attention.Parameters.init(.fromBackend(.vanilla)),
        ).rename(.{ .q = .s }).merge(.{ .d_out_proj = .{ .h, .hd } });

        const projected_output = self.o_proj
            .forward(attn_output.rename(.{ .d_out_proj = .d }), attn_output.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });

        return .{ projected_output, new_kv_cache };
    }
};

pub const RotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    rotary_dim: i64,

    pub fn init(head_dim: i64, theta: f32) RotaryEmbedding {
        return .{
            .rope_opts = .{
                .layout = .real_im_pass,
                .scaling = .{ .default = .{ .rope_theta = theta } },
            },
            .rotary_dim = head_dim,
        };
    }

    pub fn getCosAndSin(self: RotaryEmbedding, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } {
        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});
        const freqs = position_ids.convert(.f32).outer(inv_freq);
        const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, -1);
        return .{ emb.cos().convert(dtype), emb.sin().convert(dtype) };
    }

    fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(-1), 2);
        const x1 = x.slice(-1, .{ .start = 0, .end = half_dim });
        const x2 = x.slice(-1, .{ .start = half_dim, .end = x.dim(-1) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    pub fn applyRope(self: RotaryEmbedding, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        _ = self;
        const cos_x = cos.insertAxes(.hd, .{.h}).broad(x.shape());
        const sin_x = sin.insertAxes(.hd, .{.h}).broad(x.shape());
        return x.mul(cos_x).add(rotateHalf(x).mul(sin_x));
    }
};

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps).convert(x.dtype());
        return normalized.mul(self.weight.convert(x.dtype()).broad(x.shape()));
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor,

    pub const Buffers = zml.Bufferized(KvCache);

    pub fn init(config: Config, batch_dim: i64, max_seq_len: i64, dtype: zml.DataType, model_sharding: zml.Sharding) KvCache {
        const kv_shape = zml.Shape.init(.{
            .b = batch_dim,
            .layer = config.num_hidden_layers,
            .s = max_seq_len,
            .h = config.numKvHeads(),
            .hd = config.headDim(),
        }, dtype);
        const kv_head_sharding = model_sharding.shardableDim(kv_shape.dim(.h), .model, config.num_attention_heads);
        const sharded_kv_shape = switch (kv_head_sharding) {
            .sharded => |heads| kv_shape.setDim(.h, heads.dim).withPartitioning(.{ .h = .model }),
            .replicated => kv_shape.withPartitioning(.{ .h = .replicated }),
        };
        return .{
            .k = .fromShape(sharded_kv_shape),
            .v = .fromShape(sharded_kv_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initBuffer(kv: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !KvCache.Buffers {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, kv.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, kv.v.shape(), sharding, .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
        };
    }

    pub fn deinitBuffer(kv: *KvCache.Buffers) void {
        kv.k.deinit();
        kv.v.deinit();
        kv.layer_index.deinit();
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        return kv.k.slice(.layer, .dynSingle(kv.layer_index));
    }

    pub fn values(kv: KvCache) zml.Tensor {
        return kv.v.slice(.layer, .dynSingle(kv.layer_index));
    }

    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const k_shape = kv.k.shape().drop(.layer);
        var layer = kv.layer_index;
        if (token_index) |idx| {
            layer = layer.broad(idx.shape());
            return .{
                .k = kv.k.scatterSlices(
                    .{ .layer = layer, .s = idx },
                    new_k.convert(kv.k.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(kv.k),
                .v = kv.v.scatterSlices(
                    .{ .layer = layer, .s = idx },
                    new_v.convert(kv.v.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(kv.v),
                .layer_index = kv.layer_index,
            };
        }

        return .{
            .k = kv.k.scatterSlices(
                .{ .layer = layer },
                new_k.convert(kv.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(
                .{ .layer = layer },
                new_v.convert(kv.v.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(kv.v),
            .layer_index = kv.layer_index,
        };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{
            .k = kv.k,
            .v = kv.v,
            .layer_index = zml.Tensor.scalar(layer_index, .u32),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .k = kv.k.reuseBuffer(other.k),
            .v = kv.v.reuseBuffer(other.v),
            .layer_index = kv.layer_index.reuseBuffer(other.layer_index),
        };
    }
};
