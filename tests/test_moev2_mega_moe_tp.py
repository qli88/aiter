# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `mega_moe_tp` against legacy `MegaMoeTP.forward`, on CPU.

A `MegaMoeTP` without its constructor stands in for an instance: its small engine is a
`MegaMoeTPEngine` without its constructor (legacy's input checks and `tail_ok`, a
schedule choice without the tuned table, a forward that records its arguments) and its
big-M kernel a stub that records its calls. The same calls run through legacy `forward`
and through the path: the same steps with the same arguments, the same result and the
same errors. Also: the hook's arguments, the template's plans per bucket, and engines
over a fake instance (warm-up, weights, the optional tail, health, reset, release). The
TP GPU differential is tests/moe_dispatch/mega_moe_tp_differential.py.
"""

import types
from unittest import mock

import pytest
import torch

pytest.importorskip("flydsl")

from moe_dispatch.differential import moev2_mode

from aiter import ActivationType
from aiter.moev2 import engine as eng
from aiter.moev2 import paths
from aiter.moev2.migration import mega_moe_tp as path
from aiter.moev2.segments.base import MemoryNeed
from aiter.moev2.segments.mega_moe_tp import MegaMoeTPBigM, MegaMoeTPSmall, bigm_need
from aiter.ops.flydsl import mega_moe_tp as legacy
from aiter.ops.flydsl.kernels.mega_moe_tp import mega_moe_tp as legacy_engine

H, INTER, E, K = 64, 128, 9, 3
TAIL = path.TAIL
TAIL_BF16 = (*TAIL, path.TAIL_BF16)


def _value(v):
    if isinstance(v, torch.Tensor):
        return ("T", v.dtype, tuple(v.shape), v.float().tolist())
    if isinstance(v, (tuple, list)):
        return tuple(_value(x) for x in v)
    return v


def _out(out):
    return None if out is None else ("out", out.dtype, tuple(out.shape))


def _weights(scale=0):
    return {
        "w1": torch.full((E * 2 * INTER * H // 2,), scale, dtype=torch.uint8),
        "w1_scale": torch.full((E * 2 * INTER * H // 32,), scale, dtype=torch.uint8),
        "w2": torch.full((E * H * INTER // 2,), scale, dtype=torch.uint8),
        "w2_scale": torch.full((E * H * INTER // 32,), scale, dtype=torch.uint8),
    }


class _Engine(legacy_engine.MegaMoeTPEngine):
    """The small engine without its constructor: legacy's checks, a stub launch."""

    def __init__(self, log, tp, mode, mmax, lb_min, a8, weights):
        self.log = log
        self.device = torch.device("cpu")
        self.H, self.I, self.E, self.K = H, INTER, E, K
        self.rank, self.tp, self.mode, self.ar = 0, tp, mode, mode == "ar"
        self.mmax, self.lb_min, self.a8 = mmax, lb_min, a8
        self.check = False
        self.tn_local, self.tn_eps, self.tn_gemma = True, 1e-6, True
        self.swiglu_limit = 7.0
        self.err = 0
        self.ctrl = torch.zeros(8, dtype=torch.int32)
        self.routes = torch.zeros(3, H, dtype=torch.bfloat16)
        self.y = torch.empty(mmax, H, dtype=torch.bfloat16)
        self.proutes = torch.zeros(1, H, dtype=torch.bfloat16)
        self.xg = torch.empty(100, dtype=torch.uint8)
        self._bind_weights(
            weights["w1"], weights["w1_scale"], weights["w2"], weights["w2_scale"]
        )

    def config(self, m):
        tot = m * self.tp
        if self.a8 and tot > 256:
            raise ValueError(f'act_dtype="fp8": {tot} tokens are beyond the dynamic')
        if self.lb_min and tot >= self.lb_min:
            return legacy_engine.LaunchCfg(4, lb=True)
        return legacy_engine.LaunchCfg(1 if tot <= 32 else 2, ll=True)

    def forward(self, x_local, topk_weights, topk_ids, out=None, tail=None, bf16=False):
        m = self._local_tokens(x_local, topk_weights, topk_ids)
        rows = m * self.tp if self.ar else m
        self._check_out(out, rows)
        self.log.append(
            ("small", _value(x_local), _value(topk_weights), _value(topk_ids))
            + (_out(out), _value(tail), bf16)
        )
        if m == 0:
            return x_local.new_empty((0, self.H)) if out is None else out
        if tail is not None and not self.tail_ok(m):
            raise ValueError(
                "tail: ag_rs, the LB schedule, at most TN_MAX local tokens"
            )
        y = torch.full((rows, self.H), float(len(self.log)), dtype=torch.bfloat16)
        if out is not None:
            y = out.copy_(y)
        if tail is not None:
            q, s = torch.ones(m * self.tp, self.H), torch.ones(m * self.tp)
            return (y, q, s, q * 2) if bf16 else (y, q, s)
        return y

    __call__ = forward

    def set_weights(self, w1, w1_scale, w2, w2_scale):
        self.log.append(("set_weights", w1.data_ptr()))
        super().set_weights(w1, w1_scale, w2, w2_scale)

    def prepare(self, local_tokens, tail=False, tail_bf16=False, warmup=True):
        self.log.append(("prepare", tuple(local_tokens), tail, tail_bf16, warmup))

    def poll_errors(self):
        return self.err

    def reset(self):
        self.log.append(("reset",))


class _BigM:
    """The big-M kernel's host object: records its calls."""

    device = torch.device("cpu")

    def __init__(self, log, tp):
        self.log, self.tp, self.err = log, tp, 0

    def moe(
        self, x, topk_weights, topk_ids, out=None, tail=None, local=False, gather=False
    ):
        self.log.append(
            ("bigm", _value(x), _value(topk_weights), _value(topk_ids))
            + (_out(out), _value(tail), local, gather)
        )
        rows = x.shape[0] * (self.tp if gather else 1)
        y = torch.full((rows, x.shape[1]), float(len(self.log)), dtype=torch.bfloat16)
        if out is not None:
            y = out.copy_(y)
        if tail is not None:
            return y, torch.ones(rows * self.tp, x.shape[1]), torch.ones(rows * self.tp)
        return y

    def gather_rows(self, rows, local=False):
        self.log.append(("gather_rows", _value(rows), local))
        return rows

    def set_weights(self, w1, w1_scale, w2, w2_scale):
        self.log.append(("bigm.set_weights", w1.data_ptr()))

    def poll_errors(self):
        return self.err

    def reset(self):
        self.log.append(("bigm.reset",))

    def close(self):
        self.log.append(("bigm.close",))

    def io_rows_per_kernel(self, lo, hi):
        return [lo]

    def tailq(self, tail, local, gather):
        return ("tailq", tail, local, gather)

    def precompile(self, rows, tailq):
        self.log.append(("precompile", tuple(rows), tailq))


class _Fake(legacy.MegaMoeTP):
    """An instance without the constructor (see the module docstring)."""

    def __init__(
        self,
        mode="ag_rs",
        tp=4,
        max_local=64,
        small_max=32,
        bigm_m=33,
        lb_min=64,
        a8=False,
        rank=0,
        weights=None,
    ):
        weights = weights or _weights()
        self.cfg = legacy.MegaMoeTPConfig(
            rank=rank,
            world_size=tp,
            model_dim=H,
            inter_dim=INTER,
            experts=E,
            topk=K,
            max_local_tokens=max_local,
            activation=ActivationType.Swiglu,
            comm_mode=mode,
        )
        self.log = []
        self.engine = _Engine(self.log, tp, mode, small_max, lb_min, a8, weights)
        self.engine.arena = types.SimpleNamespace(
            close=lambda: self.log.append("close"),
            storage=torch.zeros(4096, dtype=torch.uint8),
        )
        self.bigm_m = bigm_m
        self._bigm = None
        self._stub = _BigM(self.log, tp)
        self._bigm_args = {**weights, "group": None, "device": None}

    def bigm(self):
        if self._bigm is None:
            self.log.append(("bigm()",))
            self._bigm = self._stub
        return self._bigm


def _inputs(rows, seed=0, wdtype=torch.float32, idtype=torch.int32):
    g = torch.Generator().manual_seed(rows + seed)
    x = torch.randn(rows, H, generator=g).to(torch.bfloat16)
    wts = torch.rand(rows, K, generator=g).to(wdtype)
    ids = torch.randint(0, E, (rows, K), generator=g).to(idtype)
    return x, wts, ids


def _tail(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    res = torch.randn(m, H, generator=g).to(torch.bfloat16)
    return res, res.clone(), torch.randn(H, generator=g).to(torch.bfloat16)


def _gfx950():
    return mock.patch.object(path, "get_gfx_runtime", return_value="gfx950")


def _kcopy():
    return mock.patch.object(legacy, "_kcopy", lambda dst, src: dst.copy_(src))


def _both(build, call):
    """`(result or error, log)` of `call(instance)` on legacy and with the path."""
    got = []
    for mode in ("off", "mega_moe_tp"):
        mtp = build()
        with moev2_mode(mode), _gfx950(), _kcopy():
            try:
                result = _value(call(mtp))
            except Exception as exc:  # noqa: BLE001
                result = f"{type(exc).__name__}: {exc}"
        got.append((result, mtp.log))
    return got


# name: (instance, rows, forward kwargs: out, tail, bf16, input dtypes)
CASES = {
    "small": ({}, 5, {}),
    "small, zero tokens": ({}, 0, {}),
    "small, out": ({}, 5, {"out": True}),
    "small, int64 ids and bf16 weights": (
        {},
        6,
        {"wdtype": torch.bfloat16, "idtype": torch.int64},
    ),
    "small, tail": ({}, 16, {"tail": True}),
    "small, tail bf16": ({}, 16, {"tail": True, "bf16": True}),
    "small, tail, out": ({}, 16, {"tail": True, "out": True}),
    "small, tail where it does not fuse": ({}, 5, {"tail": True}),
    "small, fp32 x": ({}, 5, {"x32": True}),
    "big": ({}, 40, {}),
    "big, out": ({}, 40, {"out": True}),
    "big, int64 ids and bf16 weights": (
        {},
        40,
        {"wdtype": torch.bfloat16, "idtype": torch.int64},
    ),
    "big, tail": ({}, 40, {"tail": True}),
    "big, tail bf16": ({}, 40, {"tail": True, "bf16": True}),
    "big, fp32 x": ({}, 40, {"x32": True}),
    "big, too many tokens": ({}, 65, {}),
    "small without big-M, too many tokens": ({"bigm_m": 0}, 40, {}),
    "ar small": ({"mode": "ar"}, 8, {}),
    "ar small, out": ({"mode": "ar"}, 8, {"out": True}),
    "ar small, not a multiple of tp": ({"mode": "ar"}, 6, {}),
    "ar small, tail": ({"mode": "ar"}, 128, {"tail": True}),
    "ar big": ({"mode": "ar"}, 160, {}),
    "ar big, out": ({"mode": "ar"}, 160, {"out": True}),
    "ar big, uneven": ({"mode": "ar"}, 158, {}),
    "ar big, uneven, out": ({"mode": "ar"}, 158, {"out": True}),
    "ar big, tail": ({"mode": "ar"}, 160, {"tail": True}),
}


def _call(rows, opts, ar):
    x, wts, ids = _inputs(rows, wdtype=opts.get("wdtype", torch.float32))
    ids = ids.to(opts.get("idtype", torch.int32))
    if opts.get("x32"):
        x = x.float()
    kwargs = {}
    if opts.get("tail"):
        local = -(-rows // 4) if ar else rows
        kwargs["tail"] = _tail(local)
        kwargs["bf16"] = opts.get("bf16", False)

    def call(mtp):
        if opts.get("out"):
            kwargs["out"] = torch.zeros(rows, H, dtype=torch.bfloat16)
        return mtp.forward(x, wts, ids, **kwargs)

    return call


@pytest.mark.parametrize("name", list(CASES))
def test_forward_matches_legacy(name):
    ctor, rows, opts = CASES[name]
    (ref, ref_log), (out, v2_log) = _both(
        lambda: _Fake(**ctor), _call(rows, opts, ctor.get("mode") == "ar")
    )
    assert out == ref
    assert v2_log == ref_log
    big = name.startswith(("big", "ar big"))
    if not isinstance(ref, str):
        assert ref_log[-1][0] in (("bigm", "gather_rows") if big else ("small",))


def test_off_gfx950_stays_legacy():
    x, wts, ids = _inputs(3)
    mtp = _Fake()
    runs = []
    with (
        moev2_mode("mega_moe_tp"),
        mock.patch.object(path, "get_gfx_runtime", return_value="gfx942"),
        mock.patch.object(path, "run", side_effect=runs.append),
    ):
        mtp.forward(x, wts, ids)
    assert runs == [] and mtp.log[-1][0] == "small"


def test_hook_passes_every_argument():
    x, wts, ids = _inputs(3)
    out = torch.zeros(3, H, dtype=torch.bfloat16)
    tail = _tail(3)
    seen = []
    fake_path = types.SimpleNamespace(run=lambda call: seen.append(call) or "out")
    mtp = _Fake()
    with (
        moev2_mode("mega_moe_tp"),
        mock.patch.object(
            paths, "route_to_v2", side_effect=lambda call, entry: (entry, fake_path)[1]
        ) as route,
    ):
        assert mtp.forward(x, wts, ids, out=out, tail=tail, bf16=True) == "out"
        assert mtp(x, wts, ids) == "out"
    assert [c.args[1] for c in route.call_args_list] == ["mega_tp", "mega_tp"]
    common = {"mtp": mtp, "x": x, "topk_weights": wts, "topk_ids": ids}
    assert seen[0] == {**common, "out": out, "tail": tail, "bf16": True}
    assert seen[1] == {**common, "out": None, "tail": None, "bf16": False}
    assert mtp.log == []


def test_declined_runs_legacy():
    x, wts, ids = _inputs(3)
    mtp = _Fake()
    declined = types.SimpleNamespace(run=lambda call: paths.DECLINED)
    with (
        moev2_mode("mega_moe_tp"),
        mock.patch.object(paths, "route_to_v2", return_value=declined),
    ):
        mtp.forward(x, wts, ids)
    assert [s[0] for s in mtp.log] == ["small"]


# The template.


def test_template_plans_split_at_the_switch():
    template = path.MegaMoeTPTemplate(_Fake(bigm_m=40))
    small = template.resolve(32, False)
    default = "  [] (source=default)"
    assert small.description == (
        "mega_moe_tp.small(capacity=32, comm_dtype=fp8, comm_mode=ag_rs)" + default
    )
    assert small.stateful == (template.small,) and small.graph_capturable
    assert small.output_lifetime == "until_next_forward" and small.zero_tokens_ok
    mixed = template.resolve(64, False)
    assert mixed.description == (
        "33-39 tokens: mega_moe_tp.small(capacity=32, comm_dtype=fp8, comm_mode=ag_rs)"
        f"{default}; 40-64 tokens: mega_moe_tp.bigm(capacity=64, comm_dtype=fp8, "
        f"comm_mode=ag_rs, threshold=40){default}"
    )
    assert mixed.stateful == (template.small, template.bigm)
    assert not mixed.graph_capturable
    with pytest.raises(NotImplementedError, match="above the instance's 64"):
        template.resolve(128, False)
    assert small.identity != template.resolve(16, False).identity
    assert path.MegaMoeTPTemplate(_Fake(bigm_m=40)).resolve(32, False).identity == (
        small.identity
    )
    other = path.MegaMoeTPTemplate(_Fake(bigm_m=40, lb_min=256))
    assert other.resolve(32, False).identity != small.identity
    every_row = _Fake(bigm_m=40)
    every_row.engine.tn_local = False
    every_row_plan = path.MegaMoeTPTemplate(every_row).resolve(32, False)
    assert every_row_plan.identity != small.identity
    rank3 = path.MegaMoeTPTemplate(_Fake(bigm_m=40, rank=3))
    for bucket in (32, 64):
        assert rank3.resolve(bucket, False).identity == (
            template.resolve(bucket, False).identity
        ), "every rank resolves the same plan"
    assert template.resolve(1, False).optional_side_inputs == frozenset()
    tailed = path.MegaMoeTPTemplate(_Fake(), tail=True).resolve(1, False)
    assert tailed.optional_side_inputs == frozenset(TAIL)
    bf16 = path.MegaMoeTPTemplate(_Fake(), tail_bf16=True).resolve(1, False)
    assert bf16.optional_side_inputs == frozenset(TAIL_BF16)
    assert bf16.identity != tailed.identity


def test_template_plans_without_big_m():
    template = path.MegaMoeTPTemplate(_Fake(bigm_m=0, small_max=64))
    plan = template.resolve(64, False)
    assert plan.description.startswith("mega_moe_tp.small(capacity=64")
    assert plan.graph_capturable
    with pytest.raises(NotImplementedError, match="above the instance's 64"):
        template.resolve(128, False)


def test_template_plans_in_the_ar_mode():
    template = path.MegaMoeTPTemplate(_Fake(mode="ar", bigm_m=40))
    plan = template.resolve(256, False)
    assert plan.description.startswith("129-156 tokens: mega_moe_tp.small")
    assert "; 157-256 tokens: mega_moe_tp.bigm" in plan.description
    assert template._small_locals(129, 156) == list(range(33, 40))


@pytest.mark.parametrize(
    "ctor, supported, unsupported",
    [
        ({}, [0, 1, 32, 33, 64], [65]),
        ({"mode": "ar"}, [0, 4, 128, 129, 130, 158, 256], [6, 126, 257]),
        ({"a8": True, "bigm_m": 0, "small_max": 64}, [1, 64], [65]),
        ({"a8": True, "bigm_m": 0, "small_max": 128, "max_local": 128}, [64], [65]),
        # An explicit bigm_min_tokens above the capacity: legacy runs neither on 33-47.
        ({"small_max": 32, "bigm_m": 48}, [32, 48, 64], [33, 47]),
    ],
)
def test_template_supports_what_legacy_takes(ctor, supported, unsupported):
    template = path.MegaMoeTPTemplate(_Fake(**ctor))
    for rows in supported + unsupported:
        try:
            plan = template.resolve(eng.bucket_of(rows), False)
        except NotImplementedError:
            assert rows in unsupported, rows
            continue
        assert plan.supports(rows) == (rows in supported), rows


@pytest.mark.parametrize(
    "ctor, rows, fused",
    [
        ({}, 15, False),  # 60 global tokens: the dynamic schedule
        ({}, 16, True),
        ({}, 32, True),
        ({}, 40, True),  # big-M, ag_rs
        ({"mode": "ar"}, 128, False),
        ({"mode": "ar"}, 160, False),
    ],
)
def test_template_tail_is_legacy_tail_ok(ctor, rows, fused):
    mtp = _Fake(**ctor)
    assert fused == mtp.tail_ok(mtp._local(rows))
    for kw, plain, bf16 in (
        ({"tail": True}, fused, False),
        ({"tail_bf16": True}, False, fused),
        ({"tail": True, "tail_bf16": True}, fused, fused),
    ):
        template = path.MegaMoeTPTemplate(mtp, **kw)
        assert template._supports_tail(rows, TAIL) == plain, kw
        assert template._supports_tail(rows, TAIL_BF16) == bf16, kw
        assert not template._supports_tail(rows, TAIL[:2])
        assert not template._supports_tail(rows, (*TAIL[:2], path.TAIL_BF16))


def test_template_digests():
    assert set(path.MegaMoeTPTemplate(_Fake()).digests()) == {
        "mega_moe_tp_tuned",
        "mega_moe_tp_bigm_tuned",
    }


# Engines over a fake instance (one rank).


def _engine(mtp, capture=32, **kw):
    template = path.MegaMoeTPTemplate(mtp, **kw)
    engine = eng.MoeEngine.create(template, max_tokens=64, capture_max_tokens=capture)
    engine.prepare()
    return engine


def _moe_weights(scale=0):
    w = _weights(scale)
    return eng.MoeWeights(w["w1"], w["w2"], w["w1_scale"], w["w2_scale"])


@pytest.mark.parametrize(
    "kw, flags",
    [
        ({}, (False, False)),
        ({"tail": True}, (True, False)),
        ({"tail_bf16": True}, (False, True)),
        ({"tail": True, "tail_bf16": True}, (True, True)),
    ],
)
def test_capture_prepares_the_instance_once(kw, flags):
    mtp = _Fake(tp=1, bigm_m=40, small_max=39)
    engine = _engine(mtp, **kw)
    assert mtp.log == []
    with mock.patch("torch.cuda.synchronize"):
        engine.capture()
    small = [s for s in mtp.log if s[0] == "prepare"]
    assert small == [("prepare", tuple(range(1, 33)), *flags, True)]
    # Big-M is reachable (max_local_tokens 64): legacy's prepare sets it up and warms it.
    names = [s[0] for s in mtp.log]
    assert names[:3] == ["prepare", "bigm()", "precompile"]
    assert names[-1] == ("gather_rows" if flags[1] else "bigm")


def test_a_second_engine_prepares_its_new_counts():
    mtp = _Fake(tp=1, bigm_m=40, small_max=39)
    template = path.MegaMoeTPTemplate(mtp)

    def capture(n):
        engine = eng.MoeEngine.create(template, max_tokens=n, capture_max_tokens=n)
        engine.prepare()
        with mock.patch("torch.cuda.synchronize"):
            engine.capture()
        return [s[1] for s in mtp.log if s[0] == "prepare"]

    assert capture(16) == [tuple(range(1, 17))]
    assert capture(16) == [tuple(range(1, 17))]
    assert capture(32) == [tuple(range(1, 17)), tuple(range(1, 33))]


def test_capture_prepares_only_the_counts_the_small_engine_runs():
    mtp = _Fake(tp=1, a8=True, bigm_m=0, small_max=512, max_local=512)
    engine = eng.MoeEngine.create(
        path.MegaMoeTPTemplate(mtp), max_tokens=512, capture_max_tokens=512
    )
    engine.prepare()
    with mock.patch("torch.cuda.synchronize"):
        engine.capture()
    assert [s for s in mtp.log if s[0] == "prepare"] == [
        ("prepare", tuple(range(1, 257)), False, False, True)
    ]


def test_a_captured_bucket_cannot_reach_big_m():
    with pytest.raises(NotImplementedError, match="not graph-capturable"):
        _engine(_Fake(tp=1), capture=64)


def test_engine_binds_the_layer_weights_and_runs_legacy():
    layer_a = _weights()
    mtp = _Fake(tp=1, weights=layer_a)
    engine = _engine(mtp, capture=0)
    x, wts, ids = _inputs(5)
    same = eng.MoeWeights(
        layer_a["w1"], layer_a["w2"], layer_a["w1_scale"], layer_a["w2_scale"]
    )
    engine.forward(x, ids, wts, weights=same)
    assert [s[0] for s in mtp.log] == ["small"]
    layer_b = _moe_weights(1)
    y = engine.forward(x, ids, wts, weights=layer_b)
    assert [s[0] for s in mtp.log[1:]] == ["set_weights", "small"]
    assert mtp.engine.w1.data_ptr() == layer_b.w1.data_ptr()
    assert y.shape == (5, H)
    engine.forward(x, ids, wts, weights=layer_b)
    assert [s[0] for s in mtp.log[3:]] == ["small"]
    x40, wts40, ids40 = _inputs(40)
    engine.forward(x40, ids40, wts40, weights=layer_b)
    assert [s[0] for s in mtp.log[4:]] == ["bigm()", "bigm"]
    engine.forward(x40, ids40, wts40, weights=_moe_weights(2))
    assert [s[0] for s in mtp.log[6:]] == ["set_weights", "bigm.set_weights", "bigm"]
    bad = eng.MoeWeights(
        torch.zeros(3, dtype=torch.uint8),
        layer_b.w2,
        layer_b.w1_scale,
        layer_b.w2_scale,
    )
    with pytest.raises(ValueError, match="weights.w1"):
        engine.forward(x, ids, wts, weights=bad)


def test_engine_serves_the_tail_where_legacy_fuses_it():
    mtp = _Fake(tp=1, lb_min=16)
    engine = _engine(mtp, capture=0, tail=True)
    w = _moe_weights()
    assert engine.supports(16, side_inputs=TAIL) and engine.supports(40, TAIL)
    assert not engine.supports(15, side_inputs=TAIL) and engine.supports(15)
    assert not engine.supports(16, side_inputs=TAIL[:2])
    x, wts, ids = _inputs(16)
    side = dict(zip(TAIL, _tail(16)))
    out = torch.zeros(16, H, dtype=torch.bfloat16)
    y, q, scale = engine.forward(x, ids, wts, weights=w, side_inputs=side, out=out)
    assert y is out and q.shape == (16, H) and scale.shape == (16,)
    assert mtp.log[-1][5] == _value(tuple(side[n] for n in TAIL))
    plain = engine.forward(x, ids, wts, weights=w)
    assert isinstance(plain, torch.Tensor)
    x15, wts15, ids15 = _inputs(15)
    with pytest.raises(ValueError, match="tail: ag_rs, the LB schedule"):
        engine.forward(
            x15, ids15, wts15, weights=w, side_inputs=dict(zip(TAIL, _tail(15)))
        )
    with pytest.raises(ValueError, match="the fused tail needs"):
        engine.forward(x, ids, wts, weights=w, side_inputs={"res_in": side["res_in"]})
    untailed = _engine(_Fake(tp=1, lb_min=16), capture=0)
    assert not untailed.supports(16, side_inputs=TAIL)
    with pytest.raises(ValueError, match="not read"):
        untailed.forward(x, ids, wts, weights=w, side_inputs=side)


def test_engine_picks_the_tail_kind_per_call():
    mtp = _Fake(tp=1, lb_min=16)
    engine = _engine(mtp, capture=0, tail=True, tail_bf16=True)
    assert engine.supports(16, TAIL) and engine.supports(16, TAIL_BF16)
    x, wts, ids = _inputs(16)
    side = dict(zip(TAIL, _tail(16)))
    w = _moe_weights()
    assert len(engine.forward(x, ids, wts, weights=w, side_inputs=side)) == 3
    assert mtp.log[-1][-1] is False
    bf16 = {**side, path.TAIL_BF16: True}
    assert len(engine.forward(x, ids, wts, weights=w, side_inputs=bf16)) == 4
    assert mtp.log[-1][-1] is True
    with pytest.raises(ValueError, match="the fused tail needs"):
        engine.forward(x, ids, wts, weights=w, side_inputs={path.TAIL_BF16: True})
    with pytest.raises(ValueError, match="pass True or omit it"):
        engine.forward(x, ids, wts, weights=w, side_inputs={**side, path.TAIL_BF16: 0})
    plain_only = _engine(_Fake(tp=1, lb_min=16), capture=0, tail=True)
    assert not plain_only.supports(16, TAIL_BF16)
    with pytest.raises(ValueError, match="not read"):
        plain_only.forward(x, ids, wts, weights=w, side_inputs=bf16)
    bf16_only = _engine(_Fake(tp=1, lb_min=16), capture=0, tail_bf16=True)
    assert bf16_only.supports(16, TAIL_BF16) and not bf16_only.supports(16, TAIL)


def test_engine_memory_is_the_instance_s():
    mtp = _Fake(tp=1)
    small = MegaMoeTPSmall.adopt(mtp).workspace_bytes()
    assert small.symmetric == 4096
    assert small.device == 8 * 4 + (3 + mtp.engine.mmax + 1) * H * 2 + 100
    need = bigm_need(mtp.cfg)
    assert need.device > 0 and need.symmetric % 256 == 0
    assert MegaMoeTPBigM.adopt(mtp).workspace_bytes() == need
    engine = _engine(mtp, capture=0)
    assert engine.memory() == small + need, "each segment once, over every bucket"
    no_big = _Fake(tp=1, bigm_m=0, small_max=64)
    assert MegaMoeTPBigM.adopt(no_big).workspace_bytes() == MemoryNeed()
    assert (
        _engine(no_big, capture=0).memory()
        == MegaMoeTPSmall.adopt(no_big).workspace_bytes()
    )
    below = _Fake(tp=1, max_local=128, small_max=64, bigm_m=65)
    eager = _engine(below, capture=0)
    assert eager.memory() == MegaMoeTPSmall.adopt(below).workspace_bytes()
    captured = _engine(_Fake(tp=1, max_local=128, small_max=64, bigm_m=65))
    assert captured.memory() == eager.memory() + bigm_need(
        below.cfg
    ), "capture() sets big-M up when the instance reaches it"
    budget = small.total + need.total - 1
    template = path.MegaMoeTPTemplate(_Fake(tp=1))
    over = eng.MoeEngine.create(template, max_tokens=64, memory_budget=budget)
    with pytest.raises(eng.MemoryBudgetExceeded, match="mega_moe_tp.bigm"):
        over.prepare()


def test_engine_health_reset_release():
    mtp = _Fake(tp=1)
    engine = _engine(mtp, capture=0)
    assert engine.health().ok
    engine.reset()
    assert mtp.log == [("reset",)]  # no big-M kernel yet
    x40, wts40, ids40 = _inputs(40)
    engine.forward(x40, ids40, wts40, weights=_moe_weights())
    engine.reset()
    assert mtp.log[-2:] == [("reset",), ("bigm.reset",)]
    mtp._stub.err = 1
    health = engine.health()
    assert not health.ok and "big-M wait on a peer gave up" in health.reasons[0]
    mtp._stub.err = 0
    other = _Fake(tp=1)
    engine2 = _engine(other, capture=0)
    other.engine.err = legacy_engine._ERR_NAMES[0][0] | legacy_engine._ERR_NAMES[3][0]
    health = engine2.health()
    assert not health.ok
    assert "a wait inside the kernel timed out (flag|reduce)" in health.reasons[0]
    assert engine2.state == eng.EngineState.FAILED
    third = _Fake(tp=1)
    engine3 = _engine(third, capture=0)
    engine3.release()
    assert "close" not in third.log and ("bigm.close",) not in third.log
    both = _Fake(tp=1)
    both._bigm = both._stub
    both._stub.err, both.engine.err = 1, legacy_engine._ERR_NAMES[0][0]
    reason = MegaMoeTPSmall.adopt(both).health()
    assert reason.index("big-M wait on a peer gave up") < reason.index("timed out")


def test_a_failed_capture_warmup_runs_again():
    mtp = _Fake(tp=1, bigm_m=40, small_max=39)
    template = path.MegaMoeTPTemplate(mtp)
    template.resolve(32, True)
    with (
        mock.patch.object(
            mtp.engine, "prepare", side_effect=[RuntimeError("watchdog"), None]
        ) as prepare,
        mock.patch("torch.cuda.synchronize"),
    ):
        with pytest.raises(RuntimeError, match="watchdog"):
            template._warmup()
        template._warmup()
        template._warmup()
    assert prepare.call_count == 2


def test_engine_rejects_an_instance_of_another_rank():
    segment = MegaMoeTPSmall.adopt(_Fake(tp=4, rank=1))
    with pytest.raises(
        ValueError, match="rank 1 of 4; the engine's group has rank 0 of 4"
    ):
        segment.prepare(eng.EngineContext(None, 0, 4, None, None))
    segment.prepare(eng.EngineContext(None, 1, 4, None, None))


def test_segments_adopt_one_instance():
    mtp = _Fake()
    small, bigm = MegaMoeTPSmall.adopt(mtp), MegaMoeTPBigM.adopt(mtp)
    assert small.mtp is bigm.mtp is mtp
    assert not small.serves(32) and bigm.serves(33)
    small.release()
    small.abort()
    assert mtp.log == []


def test_path_registration():
    assert "mega_tp" in paths.ENTRIES
    registered = paths.registered()["mega_moe_tp"]
    assert registered is path.PATH and registered.entry == "mega_tp"
    assert not registered.graduated
    assert registered.dispatch_record is None and not registered.segments
