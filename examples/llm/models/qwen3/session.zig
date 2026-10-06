const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled_model: *inference.CompiledModel,
    prefill: inference.KernelRunner,
    decode: inference.KernelRunner,
    layer_index_buffers: []zml.Buffer,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    tokenizer: zml.tokenizer.Tokenizer,
    generated_token_slice: zml.Slice,
    seqlen: u32,
    im_end_token_id: u32,
    end_of_text_token_id: u32,
    special_tokens: model.Model.SpecialTokens,
    think_start: ?u32,
    think_end: ?u32,
    enable_thinking: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        const inner = &compiled_model.loaded_model.inner;

        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, compiled_model.params.shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        const num_layers: usize = @intCast(inner.config.num_hidden_layers);
        const layer_index_buffers = try allocator.alloc(zml.Buffer, num_layers);
        errdefer allocator.free(layer_index_buffers);
        var initialized_layer_index_buffers: usize = 0;
        errdefer for (layer_index_buffers[0..initialized_layer_index_buffers]) |*buffer| buffer.deinit();
        for (layer_index_buffers, 0..) |*buffer, i| {
            buffer.* = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(i)), .u32);
            initialized_layer_index_buffers += 1;
        }

        const generated_token_slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));
        errdefer generated_token_slice.free(allocator);

        var prefill = try inference.KernelRunner.init(allocator, &compiled_model.prefill, model_buffers);
        errdefer prefill.deinit(allocator);
        var decode = try inference.KernelRunner.init(allocator, &compiled_model.decode, model_buffers);
        errdefer decode.deinit(allocator);

        const special_tokens = inner.special_tokens;
        const think_start = tokenizer.tokenId("<think>");
        const think_end = tokenizer.tokenId("</think>");
        if (!inner.enable_thinking and (think_start == null or think_end == null)) {
            return error.ThinkTokensNotFound;
        }

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .compiled_model = compiled_model,
            .prefill = prefill,
            .decode = decode,
            .layer_index_buffers = layer_index_buffers,
            .kv_cache_buffers = kv_cache_buffers,
            .rng_buffers = rng_buffers,
            .tokenizer = tokenizer,
            .generated_token_slice = generated_token_slice,
            .seqlen = compiled_model.params.seqlen,
            .im_end_token_id = tokenizer.tokenId("<|im_end|>") orelse special_tokens.im_end_token_id,
            .end_of_text_token_id = tokenizer.tokenId("<|endoftext|>") orelse special_tokens.end_of_text_token_id,
            .special_tokens = special_tokens,
            .think_start = think_start,
            .think_end = think_end,
            .enable_thinking = inner.enable_thinking,
        };
    }

    pub fn deinit(self: *Session) void {
        self.prefill.deinit(self.allocator);
        self.decode.deinit(self.allocator);
        for (self.layer_index_buffers) |*buffer| buffer.deinit();
        self.allocator.free(self.layer_index_buffers);
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return self.tokenizeChat(allocator, prompt, true);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return self.tokenizeChat(allocator, prompt, false);
    }

    fn tokenizeChat(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8, is_first_turn: bool) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, .{
            .special_tokens = self.special_tokens,
            .is_first_turn = is_first_turn,
            .empty_think_block = if (self.enable_thinking) null else .{
                .start = self.think_start.?,
                .end = self.think_end.?,
            },
        });
    }

    fn isStopToken(self: *const Session, token_id: u32) bool {
        return token_id == self.im_end_token_id or token_id == self.end_of_text_token_id;
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const prefill_tokens_shape = zml.Shape.init(.{ .b = 1, .s = self.seqlen }, .u32);
        const prefill_tokens_slice = try zml.Slice.alloc(self.allocator, prefill_tokens_shape);
        defer prefill_tokens_slice.free(self.allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        const replicated_sharding: zml.Sharding = .replicated;

        var prefill_tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer prefill_token_index_buffer.deinit();

        inference.run(&self.prefill, .{
            .io = self.io,
            .tokens_buf = &prefill_tokens_buffer,
            .token_index_buf = &prefill_token_index_buffer,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .rng_buffers = &self.rng_buffers,
        }, self.layer_index_buffers);

        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);
        const generated_token = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
        self.generated_token_slice.items(u32)[0] = generated_token;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const out_tokens_buffer: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);
        const replicated_sharding: zml.Sharding = .replicated;

        var current_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice, replicated_sharding);
        defer current_token_buffer.deinit();

        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(all_tokens.items.len)), .u32);
        defer token_index_buffer.deinit();

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (self.isStopToken(token_id)) break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            // Dim the reasoning section (only emitted when thinking is enabled).
            if (self.think_start) |think_start| if (token_id == think_start) {
                try stdout.writeAll("\x1b[2m");
            };
            try stdout.writeAll(token);
            if (self.think_end) |think_end| if (token_id == think_end) {
                try stdout.writeAll("\x1b[0m");
            };
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            inference.run(&self.decode, .{
                .io = self.io,
                .tokens_buf = &current_token_buffer,
                .token_index_buf = &token_index_buffer,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
            }, self.layer_index_buffers);

            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();
    }
};

const ChatPromptOptions = struct {
    special_tokens: model.Model.SpecialTokens,
    is_first_turn: bool,
    empty_think_block: ?struct { start: u32, end: u32 },
};

/// ChatML template used by Qwen3:
///   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n[<think>\n\n</think>\n\n]
/// Follow-up turns first close the previous assistant turn (`<|im_end|>\n`).
fn tokenizeChatPrompt(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    prompt: []const u8,
    opts: ChatPromptOptions,
) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenId("<|im_start|>") orelse opts.special_tokens.im_start_token_id;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse opts.special_tokens.im_end_token_id;

    const newline = try encoder.encodeAlloc(allocator, "\n");
    defer allocator.free(newline);

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 32);
    errdefer tokens.deinit(allocator);

    if (!opts.is_first_turn) {
        try tokens.append(allocator, im_end);
        try tokens.appendSlice(allocator, newline);
    }

    try tokens.append(allocator, im_start);
    try appendEncoded(allocator, &encoder, &tokens, "user\n");
    try appendEncoded(allocator, &encoder, &tokens, prompt);
    try tokens.append(allocator, im_end);
    try tokens.appendSlice(allocator, newline);
    try tokens.append(allocator, im_start);
    try appendEncoded(allocator, &encoder, &tokens, "assistant\n");

    if (opts.empty_think_block) |think| {
        // <think> and </think> are single special tokens, so they are inserted by id, not encoded from text.
        try tokens.append(allocator, think.start);
        try appendEncoded(allocator, &encoder, &tokens, "\n\n");
        try tokens.append(allocator, think.end);
        try appendEncoded(allocator, &encoder, &tokens, "\n\n");
    }

    return tokens.toOwnedSlice(allocator);
}

fn appendEncoded(allocator: std.mem.Allocator, encoder: anytype, tokens: *std.ArrayList(u32), text: []const u8) !void {
    const encoded = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(encoded);
    try tokens.appendSlice(allocator, encoded);
}
