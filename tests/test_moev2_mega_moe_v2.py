# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `mega_moe_v2` against legacy `MegaMoEV2.forward`, on CPU.

A `MegaMoEV2` without its constructor stands in for an instance: its quant, bundle-entry
choice, compact prepare, stage 1 and stage 2 are stubs that record their arguments. The
same calls run through legacy `forward` / `forward_prequant` and through the path: the
same steps with the same arguments, the same result and the same errors. Also: the
hook's arguments, `bind`, the template's instances and plans, and engines over a fake
instance. The EP8 GPU differential is tests/moe_dispatch/mega_moe_v2_differential.py.
"""

import dataclasses
import types
from unittest import mock

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("mori")

from moe_dispatch.differential import moev2_mode

from aiter.moev2 import engine as eng
from aiter.moev2 import paths
from aiter.moev2.migration import mega_moe_v2 as path
from aiter.moev2.segments import mega_moe_v2 as seg_mod
from aiter.moev2.segments.base import MemoryNeed
from aiter.moev2.segments.mega_moe_v2 import MegaMoEV2Segment
from aiter.ops.flydsl.kernels.mega_moe import mega_moe_v2 as legacy

MODEL, INTER, TOPK = 64, 32, 2
CUS = 256


class _Stream:
    cuda_stream = 1234


def _value(v):
    if isinstance(v, torch.Tensor):
        return ("T", v.dtype, tuple(v.shape), v.tolist())
    if isinstance(v, (tuple, list)):
        return tuple(_value(x) for x in v)
    if hasattr(v, "value") and type(v).__name__ == "Stream":
        return ("fx.Stream", v.value)
    return v


class _Fake(legacy.MegaMoEV2):
    """An instance without the constructor: the steps record their arguments."""

    def __init__(self, fixed=True, mtpr=16, mask=False, epr=4, rank=0, ctor=None):
        self.mtpr, self._s1_fixed_slot, self._combine_mask = mtpr, fixed, mask
        self.epr, self.model_dim, self.inter_dim = epr, MODEL, INTER
        self.rank, self.dev = rank, torch.device("cpu")
        self.ctor = ctor or {}
        self._bundle_plan = types.SimpleNamespace(
            mtpr=mtpr,
            fixed_slot_dispatch=fixed,
            entries=(
                types.SimpleNamespace(config=types.SimpleNamespace(p2p_quant="none")),
            ),
        )
        self._s1_quant_x = torch.arange(mtpr * MODEL, dtype=torch.float32).view(
            mtpr, MODEL
        )
        self._s1_quant_scale = torch.arange(mtpr * 2, dtype=torch.float32).view(mtpr, 2)
        self.log = []
        self.preloads = 0

    def preload_aot_bundles(self):
        self.preloads += 1

    def quantize(self, x_bf16):
        self.log.append(("quantize", _value(x_bf16)))
        return x_bf16.float() * 2, x_bf16.float()[:, :2]

    def _select_config(self, tokens):
        self.log.append(("select", tokens))
        return types.SimpleNamespace(stage1=("stage1", tokens), stage2="stage2")

    def _run_compact_prepare(
        self, topk_ids, cur_tok, config, stream, *, quant_input=None
    ):
        self.log.append(
            (
                "prepare",
                _value(topk_ids),
                cur_tok,
                config,
                _value(stream),
                _value(quant_input),
            )
        )

    def _run_fused_stage1(
        self, x, wts, scales, topk_ids, stream=None, config=None, *, prepared=False
    ):
        self.log.append(
            (
                "stage1",
                _value(x),
                _value(wts),
                _value(scales),
                _value(topk_ids),
                _value(stream),
                config,
                prepared,
            )
        )

    def _run_stage2(self, run_tokens, stream, slice_output, config, combine_ids=None):
        self.log.append(
            (
                "stage2",
                run_tokens,
                _value(stream),
                slice_output,
                config,
                _value(combine_ids),
            )
        )
        return torch.full((run_tokens, MODEL), float(len(self.log)))


def _inputs(tokens, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, MODEL, generator=g).to(dtype)
    wts = torch.rand(tokens, TOPK, generator=g)
    ids = torch.randint(0, 8, (tokens, TOPK), generator=g, dtype=torch.int32)
    return x, wts, ids


def _gfx950():
    return mock.patch.object(path, "get_gfx_runtime", return_value="gfx950")


def _both(build, call):
    """`(result or error, log)` of `call(instance)` on legacy and with the path."""
    out = []
    for mode in ("off", "mega_moe_v2"):
        mega = build()
        with moev2_mode(mode), _gfx950():
            try:
                result = call(mega)
            except Exception as exc:  # noqa: BLE001
                result = f"{type(exc).__name__}: {exc}"
        out.append((result if isinstance(result, str) else result.tolist(), mega.log))
    return out


CASES = {
    "fixed": ({"fixed": True}, 5, {}),
    "fixed, zero tokens": ({"fixed": True}, 0, {}),
    "fixed, mask": ({"fixed": True}, 5, {"mask_invalid_slots": True}),
    "fixed, mask by default": ({"fixed": True, "mask": True}, 5, {}),
    "fixed, mask off over the default": (
        {"fixed": True, "mask": True},
        5,
        {"mask_invalid_slots": False},
    ),
    "fixed, full output, stream": (
        {"fixed": True},
        5,
        {"slice_output": False, "stream": _Stream()},
    ),
    "compact": ({"fixed": False}, 7, {"stream": _Stream()}),
    "compact, mask": (
        {"fixed": False},
        7,
        {"stream": _Stream(), "mask_invalid_slots": True},
    ),
    "compact, at the capacity": ({"fixed": False}, 16, {"stream": _Stream()}),
    "too many tokens": ({"fixed": True}, 17, {}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_forward_matches_legacy(name):
    ctor, tokens, kwargs = CASES[name]
    x, wts, ids = _inputs(tokens)
    (ref, ref_log), (out, v2_log) = _both(
        lambda: _Fake(**ctor), lambda m: m.forward(x, wts, ids, **kwargs)
    )
    assert out == ref
    assert v2_log == ref_log
    if not isinstance(ref, str):
        assert ref_log[-1][0] == "stage2"


@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize(
    "kwargs", [{}, {"mask_invalid_slots": True}, {"slice_output": False}]
)
def test_forward_prequant_matches_legacy(fixed, kwargs):
    x, wts, ids = _inputs(6)
    x_q, scales = x.float() * 3, x.float()[:, :2]
    (ref, ref_log), (out, v2_log) = _both(
        lambda: _Fake(fixed=fixed),
        lambda m: m.forward_prequant(x_q, scales, wts, ids, **kwargs),
    )
    assert out == ref and v2_log == ref_log
    assert [step[0] for step in ref_log] == ["select", "stage1", "stage2"]


@pytest.mark.parametrize(
    "bad",
    ["x fp32", "x non-contiguous", "wts fp16", "ids int64", "ids non-contiguous"],
)
def test_forward_errors_match_legacy(bad):
    x, wts, ids = _inputs(4)
    if bad == "x fp32":
        x = x.float()
    elif bad == "x non-contiguous":
        x = torch.cat([x, x], dim=1)[:, ::2]
    elif bad == "wts fp16":
        wts = wts.half()
    elif bad == "ids int64":
        ids = ids.long()
    else:
        ids = torch.cat([ids, ids], dim=1)[:, ::2]
    (ref, ref_log), (out, v2_log) = _both(_Fake, lambda m: m.forward(x, wts, ids))
    assert isinstance(ref, str) and ref.startswith("ValueError") and out == ref
    assert ref_log == v2_log == []


def test_off_gfx950_stays_legacy():
    x, wts, ids = _inputs(3)
    mega = _Fake()
    runs = []
    with moev2_mode("mega_moe_v2"), mock.patch.object(
        path, "get_gfx_runtime", return_value="gfx942"
    ), mock.patch.object(path, "run", side_effect=runs.append):
        mega.forward(x, wts, ids)
    assert runs == [] and mega.log[-1][0] == "stage2"


def test_hook_passes_every_argument():
    x, wts, ids = _inputs(3)
    scales = torch.ones(3, 2)
    stream = _Stream()
    seen = []
    fake_path = types.SimpleNamespace(
        name="mega_moe_v2", entry="mega_v2", run=lambda call: seen.append(call) or "out"
    )
    mega = _Fake()
    paths.reset_outcomes()
    with moev2_mode("mega_moe_v2"), mock.patch.object(
        paths, "route_to_v2", side_effect=lambda call, entry: (entry, fake_path)[1]
    ) as route:
        assert mega.forward(x, wts, ids, stream=stream, slice_output=False) == "out"
        assert (
            mega.forward_prequant(x, scales, wts, ids, mask_invalid_slots=True) == "out"
        )
    assert [c.args[1] for c in route.call_args_list] == ["mega_v2", "mega_v2"]
    assert paths.outcomes() == {("mega_v2", "mega_moe_v2", "served"): 2}
    common = {"mega": mega, "wts": wts, "topk_ids": ids}
    assert seen[0] == {
        **common,
        "prequant": False,
        "x": x,
        "scales": None,
        "stream": stream,
        "slice_output": False,
        "mask_invalid_slots": None,
    }
    assert seen[1] == {
        **common,
        "prequant": True,
        "x": x,
        "scales": scales,
        "stream": None,
        "slice_output": True,
        "mask_invalid_slots": True,
    }
    assert mega.log == []


def test_declined_runs_legacy():
    x, wts, ids = _inputs(3)
    mega = _Fake()
    paths.reset_outcomes()
    declined = types.SimpleNamespace(
        name="mega_moe_v2", entry="mega_v2", run=lambda call: paths.DECLINED
    )
    with moev2_mode("mega_moe_v2"), mock.patch.object(
        paths, "route_to_v2", return_value=declined
    ):
        mega.forward(x, wts, ids)
    assert [s[0] for s in mega.log] == ["quantize", "select", "stage1", "stage2"]
    assert paths.outcomes() == {("mega_v2", "mega_moe_v2", "declined"): 1}


def _weights(epr, *, dtype=torch.uint8, short=0):
    n = epr * 2 * INTER * MODEL
    return (
        torch.zeros(n // 2, dtype=dtype),
        torch.zeros(n // 32 - short, dtype=dtype),
        torch.zeros(epr * MODEL * INTER // 2, dtype=dtype),
        torch.zeros(epr * MODEL * INTER // 32, dtype=dtype),
    )


def test_bind_swaps_the_weights():
    mega = _Fake(epr=4)
    w1, w1_scale, w2, w2_scale = _weights(4, dtype=torch.float8_e4m3fn)
    mega.bind(w1, w1_scale, w2, w2_scale)
    assert mega._s1_w1.dtype == torch.uint8 and mega._s1_w1.data_ptr() == w1.data_ptr()
    assert mega._s1_w1_scale.data_ptr() == w1_scale.data_ptr()
    assert mega.w2 is w2 and mega.w2_scale is w2_scale


def test_bind_takes_padded_scales():
    mega = _Fake(epr=4)
    w1, w1_scale, w2, w2_scale = _weights(4)
    w1_scale = torch.zeros(w1_scale.numel() + 64, dtype=torch.uint8)
    mega.bind(w1, w1_scale, w2, w2_scale)
    assert mega._s1_w1_scale.data_ptr() == w1_scale.data_ptr()


@pytest.mark.parametrize(
    "bad, match",
    [
        ("short", "w1_scale: need"),
        ("long", "w2: need a contiguous tensor of 4096 one-byte"),
        ("non-contiguous", "w1: need a contiguous"),
        ("two-byte", "w1: need"),
        ("device", "w2: need"),
    ],
)
def test_bind_raises_instead_of_copying(bad, match):
    mega = _Fake(epr=4)
    w1, w1_scale, w2, w2_scale = _weights(4, short=1 if bad == "short" else 0)
    if bad == "non-contiguous":
        w1 = torch.zeros(2 * w1.numel(), dtype=torch.uint8)[::2]
    if bad == "two-byte":
        w1 = torch.zeros(w1.numel(), dtype=torch.float16)
    if bad == "device":
        w2 = w2.to("meta")
    if bad == "long":
        w2 = torch.zeros(w2.numel() + 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match=match):
        mega.bind(w1, w1_scale, w2, w2_scale)
    assert not hasattr(mega, "_s1_w1")


def _meta_weights(epr, model=7168, inter=3072):
    def meta(*shape):
        return torch.empty(shape, dtype=torch.uint8, device="meta")

    return eng.MoeWeights(
        meta(epr, 2 * inter, model // 2),
        meta(epr, model, inter // 2),
        meta(epr * 2 * inter * model // 32),
        meta(epr * model * inter // 32),
    )


def _v4_template(max_tokens, shared=False, **kw):
    return path.MegaMoEV2Template(
        _meta_weights(48 + shared),
        world_size=8,
        model_dim=7168,
        inter_dim=3072,
        experts=384,
        topk=6,
        max_tokens_per_rank=max_tokens,
        swiglu_limit=10.0,
        fuse_shared_expert=shared,
        **kw,
    )


@pytest.mark.parametrize(
    "max_tokens, shared, capacities",
    [
        (8192, False, (128, 8192)),
        (1024, False, (128, 1024)),
        (128, False, (128,)),
        (64, False, (64,)),
        (8192, True, (8192,)),  # 49 local experts: no fixed-slot
    ],
)
def test_template_instances_follow_vllm(max_tokens, shared, capacities):
    template = _v4_template(max_tokens, shared)
    assert template.capacities == capacities
    spec = template.spec(capacities[-1])
    assert spec.experts == 384 + 8 * shared and spec.topk == 6 + shared


def test_template_plans():
    template = _v4_template(8192)
    plans = {b: template.resolve(b, False) for b in (1, 64, 128, 256, 8192)}
    for b in (1, 64, 128):
        assert plans[b].description.startswith(
            "mega_moe_v2(capacity=128, dispatch=fixed_slot, p2p_quant=none)"
        )
        assert plans[b].stateful[0] is plans[1].stateful[0]
    for b in (256, 8192):
        assert plans[b].description.startswith(
            "mega_moe_v2(capacity=8192, dispatch=compact, p2p_quant=fp8_blockwise_1x32)"
        )
    assert plans[1].identity != plans[256].identity
    assert plans[1].output_lifetime == "until_next_forward"
    assert plans[1].zero_tokens_ok and plans[1].graph_capturable
    with pytest.raises(NotImplementedError, match="above max_tokens_per_rank=8192"):
        template.resolve(16384, False)
    assert _v4_template(8192).resolve(1, False).identity == plans[1].identity
    masked = _v4_template(8192, mask_invalid_slots=True).resolve(1, False)
    assert masked.identity != plans[1].identity


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"max_tokens_per_rank": 1000}, "power of two"),
        ({"experts": 385}, "divisible"),
        (
            {"weights": dataclasses.replace(_meta_weights(48), w2_scale=None)},
            "w2_scale",
        ),
    ],
)
def test_template_checks(kw, match):
    args = {
        "weights": _meta_weights(48),
        "world_size": 8,
        "model_dim": 7168,
        "inter_dim": 3072,
        "experts": 384,
        "topk": 6,
        "max_tokens_per_rank": 1024,
    }
    with pytest.raises(ValueError, match=match):
        path.MegaMoEV2Template(**{**args, **kw})


def _fake_module(built):
    def ctor(**kwargs):
        mega = _Fake(
            fixed=False,
            mtpr=kwargs["max_tok_per_rank"],
            epr=kwargs["experts"] // kwargs["world_size"],
            rank=kwargs["rank"],
            ctor=kwargs,
        )
        built.append(mega)
        return mega

    return types.SimpleNamespace(MegaMoEV2=ctor)


def _cpu_weights(epr):
    w1, w1_scale, w2, w2_scale = _weights(epr)
    return eng.MoeWeights(w1, w2, w1_scale, w2_scale)


@pytest.mark.parametrize("shared", [False, True])
def test_engine_binds_and_runs(shared):
    epr = 4 + shared
    layer_a, layer_b = _cpu_weights(epr), _cpu_weights(epr)
    template = path.MegaMoEV2Template(
        layer_a,
        world_size=1,
        model_dim=MODEL,
        inter_dim=INTER,
        experts=4,
        topk=TOPK,
        max_tokens_per_rank=16,
        fuse_shared_expert=shared,
    )
    assert template.capacities == (16,)
    built = []
    appended = []

    def append(weights, ids, *, shared_expert_id, routed_per_rank):
        appended.append((shared_expert_id, routed_per_rank))
        col = torch.full((ids.shape[0], 1), shared_expert_id, dtype=ids.dtype)
        return (
            torch.cat([weights, torch.ones(ids.shape[0], 1)], 1),
            torch.cat([ids, col], 1),
        )

    with mock.patch.object(seg_mod, "_mega", lambda: _fake_module(built)), mock.patch(
        "aiter.moev2.routing.shared_route.append_shared_expert", append
    ), mock.patch.object(seg_mod, "_num_cu", return_value=CUS):
        engine = eng.MoeEngine.create(template, max_tokens=16, capture_max_tokens=16)
        engine.prepare()
        assert len(built) == 1
        mega = built[0]
        segment = template._segments[16]
        assert engine.memory() == seg_mod.instance_need(segment.spec, layer_a, CUS)
        engine.capture()
        assert mega.preloads == 1 and not mega.log
        assert appended == ([(4, 4)] if shared else [])
        appended.clear()
        assert mega.ctor["w1"] is layer_a.w1 and mega.ctor["max_tok_per_rank"] == 16
        assert mega.ctor["experts"] == 4 + shared and mega.ctor["topk"] == TOPK + shared
        x, wts, ids = _inputs(5)
        out = engine.forward(x, ids, wts, weights=layer_b)
        assert out.shape == (5, MODEL)
        assert mega._s1_w1.data_ptr() == layer_b.w1.data_ptr()
        assert mega.w2 is layer_b.w2
        steps = [s[0] for s in mega.log]
        assert steps == ["select", "prepare", "stage1", "stage2"]
        stage1_ids = mega.log[2][4]
        assert len(stage1_ids[3][0]) == TOPK + shared
        assert appended == ([(4, 4)] if shared else [])
        with pytest.raises(ValueError, match="weights.w1"):
            engine.forward(x, ids, wts, weights=_cpu_weights(epr + 1))
        engine.release()
    assert template._segments[16].mega is None


def test_instance_need_per_dispatch_and_weights():
    template = _v4_template(1024)
    fixed, compact = (template.resolve(c, False).stateful[0] for c in (128, 1024))
    assert fixed.spec.bundle_plan().fixed_slot_dispatch
    assert not compact.spec.bundle_plan().fixed_slot_dispatch
    need = seg_mod.instance_need(fixed.spec, fixed.weights, CUS)
    assert need.device > 0 and need.symmetric > 0
    assert seg_mod.instance_need(compact.spec, compact.weights, CUS) != need
    with mock.patch.object(seg_mod, "_num_cu", return_value=CUS):
        assert fixed.workspace_bytes() == need
    w = fixed.weights
    w1 = w.w1.transpose(1, 2)
    strided = dataclasses.replace(w, w1=w1)
    with_copy = seg_mod.instance_need(fixed.spec, strided, CUS)
    assert with_copy == MemoryNeed(need.device + w1.nbytes, need.symmetric)
    assert MegaMoEV2Segment.adopt(_Fake()).workspace_bytes() == MemoryNeed()


def test_engine_budget_refuses_before_building():
    built = []
    template = path.MegaMoEV2Template(
        _cpu_weights(4),
        world_size=1,
        model_dim=MODEL,
        inter_dim=INTER,
        experts=4,
        topk=TOPK,
        max_tokens_per_rank=16,
    )
    spec = template.resolve(16, False).stateful[0].spec
    need = seg_mod.instance_need(spec, template.weights, CUS)
    with mock.patch.object(
        seg_mod, "_mega", lambda: _fake_module(built)
    ), mock.patch.object(seg_mod, "_num_cu", return_value=CUS):
        engine = eng.MoeEngine.create(
            template, max_tokens=16, memory_budget=need.total - 1
        )
        with pytest.raises(eng.MemoryBudgetExceeded, match="mega_moe_v2"):
            engine.prepare()
    assert not built


def test_engine_rejects_an_instance_for_another_world():
    template = _v4_template(64)
    segment = template.resolve(1, False).stateful[0]
    ctx = eng.EngineContext(None, 0, 4, None, None)
    with pytest.raises(ValueError, match="for 8 ranks; the group has 4"):
        segment.prepare(ctx)


def test_adopted_instance_is_not_released():
    mega = _Fake()
    segment = MegaMoEV2Segment.adopt(mega)
    segment.prepare(eng.EngineContext(None, 0, 1, None, None))
    segment.release()
    assert segment.mega is mega
    assert segment.params().get("dispatch") == "fixed_slot"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")
@pytest.mark.parametrize("tokens", [0, 1, 70, 200])
def test_shared_route_matches_vllm(tokens):
    from aiter.moev2.routing import shared_route

    routed = 48
    g = torch.Generator().manual_seed(tokens)
    ids = torch.randint(0, 384, (tokens, 6), generator=g, dtype=torch.int32)
    if tokens:
        ids[0, 1] = -1
    weights = torch.rand(tokens, 6, generator=g, dtype=torch.float32).half()
    ids, weights = ids.cuda(), weights.cuda()
    sid = shared_route.shared_expert_id(3, routed)
    assert sid == 3 * 49 + 48
    got_w, got_ids = shared_route.append_shared_expert(
        weights, ids, shared_expert_id=sid, routed_per_rank=routed
    )
    want_ids = torch.cat(
        [
            torch.where(ids >= 0, ids + ids // routed, ids),
            torch.full((tokens, 1), sid, dtype=ids.dtype, device=ids.device),
        ],
        1,
    )
    want_w = torch.cat(
        [weights.float(), torch.ones(tokens, 1, device=weights.device)], 1
    )
    assert torch.equal(got_ids, want_ids) and torch.equal(got_w, want_w)
    assert got_w.dtype == torch.float32 and got_w.is_contiguous()


def test_path_registration():
    assert "mega_v2" in paths.ENTRIES
    registered = paths.registered()["mega_moe_v2"]
    assert registered is path.PATH and registered.entry == "mega_v2"
    assert not registered.graduated
    assert registered.dispatch_record is None and not registered.segments
