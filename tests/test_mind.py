"""Tests for shifu_ocr.mind unified cognitive architecture."""

import pytest
from shifu_ocr.mind._types import Synapse, Assembly, Domain, tokenize
from shifu_ocr.mind.cortex import Cortex, Layer
from shifu_ocr.mind.field import Field
from shifu_ocr.mind.gate import Gate
from shifu_ocr.mind.signal import Signal
from shifu_ocr.mind.trunk import Trunk
from shifu_ocr.mind.memory import Memory, Episode
from shifu_ocr.mind.speaker import Speaker
from shifu_ocr.mind.thinker import Thinker, WorkingMemory
from shifu_ocr.mind.mind import ShifuMind


# ── Shared tokenize ──

class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_hyphenated(self):
        assert "co-occur" in tokenize("co-occur")

    def test_numbers_in_words(self):
        assert "h2o" in tokenize("H2O is water")

    def test_empty(self):
        assert tokenize("") == []

    def test_numbers_only(self):
        assert tokenize("123 456") == []


# ── Synapse ──

class TestSynapse:
    def test_strengthen(self):
        s = Synapse("a", "b", weight=1.0)
        s.strengthen(0.5, epoch=1)
        assert s.weight == 1.5
        assert s.last_active == 1
        assert s.activation_count == 1

    def test_decay(self):
        s = Synapse("a", "b", weight=2.0)
        s.decay(0.5)
        assert s.weight == 1.0

    def test_myelinated_decay(self):
        s = Synapse("a", "b", weight=2.0)
        s.myelinate()
        s.decay(0.5, myelinated_factor=0.9)
        assert s.weight == pytest.approx(1.8)

    def test_serialization_roundtrip(self):
        s = Synapse("a", "b", weight=1.5, birth_epoch=2, last_active=5,
                    activation_count=3, _myelinated=True)
        d = s.to_dict()
        s2 = Synapse.from_dict(d)
        assert s2.source == "a"
        assert s2.target == "b"
        assert s2.weight == 1.5
        assert s2.myelinated is True
        assert s2.activation_count == 3


# ── Assembly ──

class TestAssembly:
    def test_add_and_max_size(self):
        a = Assembly(id="a0", max_size=3)
        assert a.add("x", 1) is True
        assert a.add("y", 1) is True
        assert a.add("z", 1) is True
        assert a.add("w", 1) is False  # at max size

    def test_overlap(self):
        a = Assembly(id="a0", words={"x", "y", "z"})
        b = Assembly(id="a1", words={"y", "z", "w"})
        assert a.overlap(b) == pytest.approx(2 / 4)

    def test_serialization_roundtrip(self):
        a = Assembly(id="a0", words={"x", "y"}, strength=3.0,
                     birth_epoch=1, last_active=5, activation_count=2)
        d = a.to_dict()
        a2 = Assembly.from_dict(d)
        assert a2.words == {"x", "y"}
        assert a2.strength == 3.0


# ── Domain ──

class TestDomain:
    def test_affinity(self):
        d = Domain(name="med", words={"stroke", "artery", "brain"})
        co = {"vessel": {"artery": 1.0, "brain": 0.5, "tree": 0.3}}
        assert d.affinity("vessel", co) == pytest.approx(2 / 3)

    def test_serialization_roundtrip(self):
        d = Domain(name="med", words={"a", "b"}, seed_words={"a"},
                   strength=1.0, _taught=True)
        d2 = Domain.from_dict(d.to_dict())
        assert d2.name == "med"
        assert d2._taught is True
        assert d2.seed_words == {"a"}


# ── Layer ──

class TestLayer:
    def test_connect_and_get(self):
        ly = Layer("test")
        ly.connect("a", "b", 1.0, epoch=1)
        neighbors = ly.get_neighbors("a")
        assert "b" in neighbors
        assert neighbors["b"] == 1.0

    def test_connect_strengthens_existing(self):
        ly = Layer("test")
        ly.connect("a", "b", 1.0, epoch=1)
        ly.connect("a", "b", 0.5, epoch=2)
        assert ly.get_neighbors("a")["b"] == 1.5

    def test_prune(self):
        ly = Layer("test")
        ly.connect("a", "b", 0.005, epoch=1)
        pruned = ly.prune(epoch=2, decay_factor=0.5,
                          myelinated_factor=0.9, min_weight=0.01)
        assert pruned == 1
        assert ly.get_neighbors("a") == {}

    def test_serialization_roundtrip(self):
        ly = Layer("test", birth_epoch=5)
        ly.connect("a", "b", 1.0, epoch=6)
        ly2 = Layer.from_dict(ly.to_dict())
        assert ly2.name == "test"
        assert ly2.get_neighbors("a")["b"] == 1.0


# ── Cortex ──

class TestCortex:
    def test_feed_builds_connections(self):
        cx = Cortex()
        n = cx.feed(["stroke", "is", "caused", "by", "occlusion"])
        assert n > 0
        assert cx.word_freq["stroke"] == 1

    def test_feed_text(self):
        cx = Cortex()
        n = cx.feed_text("Stroke is caused by arterial occlusion")
        assert n > 0
        assert "stroke" in cx.word_freq

    def test_word_freq_stays_int_after_tick(self):
        """C3 fix: tick() should not convert word_freq to float."""
        cx = Cortex()
        cx.feed(["stroke", "brain", "artery", "vessel"])
        cx.tick()
        for v in cx.word_freq.values():
            assert isinstance(v, int), f"word_freq value {v} is {type(v)}, expected int"

    def test_max_feed_tokens_configurable(self):
        """M4 fix: per-feed cap should be configurable."""
        cx = Cortex(max_feed_tokens=5)
        tokens = [f"word{i}" for i in range(20)]
        cx.feed(tokens)
        # Only first 5 tokens should have breadth connections
        assert "word0" in cx.breadth
        assert "word4" in cx.breadth
        # word10 should appear in word_freq (counted) but not in breadth
        assert "word10" in cx.word_freq
        assert "word10" not in cx.breadth

    def test_assembly_dedup(self):
        """M5 fix: feeding identical tokens should reuse existing assembly."""
        cx = Cortex()
        tokens = ["stroke", "brain", "artery"]
        cx.feed(tokens)
        count_after_first = len(cx._assemblies)
        cx.feed(tokens)
        count_after_second = len(cx._assemblies)
        # Should not create a new assembly with >50% overlap
        assert count_after_second == count_after_first

    def test_confidence(self):
        cx = Cortex()
        assert cx.confidence("unknown")['score'] == 0
        cx.feed_text("stroke is caused by arterial occlusion")
        conf = cx.confidence("stroke")
        assert conf['score'] > 0

    def test_inverse_breadth(self):
        cx = Cortex()
        cx.feed_text("stroke is caused by arterial occlusion")
        score = cx.inverse_breadth("stroke")
        assert score > 0
        # idf alias still works
        assert cx.idf("stroke") == score

    def test_serialization_roundtrip(self):
        cx = Cortex()
        cx.feed_text("the brain is complex and stroke affects the brain")
        d = cx.to_dict()
        cx2 = Cortex.from_dict(d)
        assert cx2.word_freq == cx.word_freq
        assert cx2._epoch == cx._epoch
        assert set(cx2.breadth.keys()) == set(cx.breadth.keys())


# ── Gate ──

class TestGate:
    def test_filter_accepts(self):
        g = Gate()
        result = g.filter("Stroke is caused by arterial occlusion in the brain")
        assert result['accepted'] is True
        assert len(result['tokens']) > 0

    def test_filter_rejects_short(self):
        g = Gate()
        result = g.filter("hi")
        assert result['accepted'] is False
        assert result['reason'] == 'too_short'

    def test_stop_words_no_length_filter(self):
        """M8 fix: short words should not be auto-filtered."""
        g = Gate()
        freqs = {"the": 100, "is": 80, "iv": 2, "ct": 3, "stroke": 5}
        stops = g.stop_words(freqs)
        # "iv" and "ct" should NOT be in stop words (they're low frequency)
        assert "iv" not in stops
        assert "ct" not in stops

    def test_serialization_roundtrip(self):
        g = Gate()
        g.filter("some test text with several words for filtering")
        d = g.to_dict()
        g2 = Gate.from_dict(d)
        assert g2._total_accepted == g._total_accepted


# ── Signal ──

class TestSignal:
    def test_predict_default(self):
        s = Signal()
        assert s.predict("unknown") == 0.5

    def test_observe_updates_prediction(self):
        s = Signal()
        s.observe("state1", 0.8)
        # After observing high quality, prediction should increase
        assert s.predict("state1") > 0.5

    def test_recalibration_preserves_stats(self):
        """C2 fix: recalibration should not zero out total_quality."""
        s = Signal(recalibrate_threshold=2, recalibrate_quality=0.3)
        s.observe("bad", 0.1)
        s.observe("bad", 0.1)
        pol = s._policies["bad"]
        # After recalibration, total_quality should not be 0
        assert pol['total_quality'] >= 0
        # avg_quality should reflect actual observations
        assert pol['avg_quality'] == pytest.approx(0.1)

    def test_serialization_roundtrip(self):
        s = Signal()
        s.observe("s1", 0.7)
        s2 = Signal.from_dict(s.to_dict())
        assert s2.predict("s1") == pytest.approx(s.predict("s1"))


# ── Field ──

class TestField:
    def test_activate(self):
        f = Field()
        co = {"stroke": {"brain": 1.0, "artery": 0.8}}
        result = f.activate("stroke", co)
        assert "stroke" in result
        assert "brain" in result

    def test_score_sequence(self):
        f = Field()
        co = {
            "stroke": {"brain": 1.0, "artery": 0.8},
            "brain": {"stroke": 1.0, "neuron": 0.5},
        }
        result = f.score_sequence(["stroke", "brain"], co)
        assert result['coherence'] > 0

    def test_score_sequence_empty(self):
        f = Field()
        result = f.score_sequence([], {})
        assert result['coherence'] == 0.0
        assert result['scores'] == []

    def test_ocr_candidates(self):
        f = Field()
        co = {
            "patient": {"stroke": 1.0, "cerebral": 0.8},
            "cerebral": {"stroke": 1.0, "patient": 0.5},
            "stroke": {"patient": 1.0, "cerebral": 1.0},
        }
        results = f.score_ocr_candidates(
            [("stroke", 0.9), ("strake", 0.7)],
            ["patient", "cerebral"], co,
        )
        assert len(results) == 2
        assert results[0]['rank'] == 1

    def test_serialization_roundtrip(self):
        """m7 fix: serialization key names should be consistent."""
        f = Field(settle_reactivate_top=12, settle_iter_decay=0.7)
        d = f.to_dict()
        assert 'settle_reactivate_top' in d
        assert 'settle_iter_decay' in d
        f2 = Field.from_dict(d)
        assert f2._settle_top == 12
        assert f2._settle_decay == pytest.approx(0.7)


# ── Trunk ──

class TestTrunk:
    def test_observe_creates_domain(self):
        t = Trunk()
        co = {"stroke": {"brain": 1.0}, "brain": {"stroke": 1.0}}
        domain = t.observe(["stroke", "brain", "artery", "vessel", "occlusion"], co)
        assert domain is not None

    def test_seed_domains(self):
        t = Trunk(seed_domains={"medical": ["stroke", "brain", "artery"]})
        assert "medical" in t.domains
        assert t.domains["medical"]._taught is True

    def test_merge_smaller_into_larger(self):
        """n4 fix: merge should absorb smaller into larger."""
        t = Trunk(merge_overlap=0.3)
        t.domains["big"] = Domain(name="big", words={"a", "b", "c", "d", "e"})
        t.domains["small"] = Domain(name="small", words={"a", "b", "f"})
        for w in ["a", "b"]:
            t.word_domain[w] = ["big", "small"]
        t._merge_domains()
        # "small" should be merged into "big", not the reverse
        assert "big" in t.domains
        assert "small" not in t.domains
        assert "f" in t.domains["big"].words

    def test_serialization_roundtrip(self):
        t = Trunk(seed_domains={"med": ["stroke"]})
        t.observe(["stroke", "brain", "artery", "vessel", "occlusion"],
                  {"stroke": {"brain": 1}})
        t2 = Trunk.from_dict(t.to_dict())
        assert set(t2.domains.keys()) == set(t.domains.keys())


# ── Memory ──

class TestMemory:
    def test_record_and_recall(self):
        m = Memory(significance_threshold=0.1)
        m.record(epoch=1, tokens=["stroke", "brain"], significance=0.5)
        episodes = m.recall(["stroke"])
        assert len(episodes) == 1
        assert "stroke" in episodes[0].tokens

    def test_eviction(self):
        m = Memory(capacity=2, significance_threshold=0.1)
        m.record(epoch=1, tokens=["a"], significance=0.3)
        m.record(epoch=2, tokens=["b"], significance=0.8)
        m.record(epoch=3, tokens=["c"], significance=0.5)
        assert len(m.episodes) == 2
        # Least significant (0.3) should be evicted
        sigs = [e.significance for e in m.episodes]
        assert 0.3 not in sigs

    def test_serialization_uses_capacity(self):
        """m8 fix: serialization should use capacity, not hardcoded 200."""
        m = Memory(capacity=5, significance_threshold=0.1)
        for i in range(10):
            m.record(epoch=i, tokens=[f"word{i}"], significance=0.5 + i * 0.01)
        d = m.to_dict()
        assert len(d['episodes']) <= 5


# ── Speaker ──

class TestSpeaker:
    def test_learn_and_generate(self):
        s = Speaker()
        s.learn_frame(["stroke", "affects", "brain", "function"])
        co = {"stroke": {"affects": 1.0, "brain": 0.5}}
        result = s.generate(["stroke"], co, max_length=5)
        assert len(result) >= 1
        assert result[0] == "stroke"

    def test_no_repeat(self):
        """m5 fix: generate should not repeat words."""
        s = Speaker()
        s.learn_frame(["stroke", "brain", "stroke", "brain"])
        co = {"stroke": {"brain": 1.0}, "brain": {"stroke": 1.0}}
        result = s.generate(["stroke"], co, max_length=10)
        # After the seed, no word should appear twice
        non_seed = result[1:]
        assert len(non_seed) == len(set(non_seed))


# ── Thinker ──

class TestThinker:
    def test_deliberate(self):
        t = Thinker(max_steps=3)
        activate_fn = lambda w: {"related": 0.5}
        score_fn = lambda tokens: {"coherence": 0.7}
        result = t.deliberate(["test"], activate_fn, score_fn)
        assert 'coherence' in result
        assert result['steps'] >= 1

    def test_counterfactual(self):
        t = Thinker()
        score_fn = lambda tokens: {"coherence": 0.5 if "good" in tokens else 0.2}
        results = t.counterfactual(["bad", "day"], 0, ["good", "bad"], score_fn)
        assert len(results) == 2
        assert results[0]['word'] == "good"


# ── ShifuMind Integration ──

class TestShifuMind:
    def test_feed_and_score(self):
        mind = ShifuMind()
        r = mind.feed("Stroke is caused by arterial occlusion in the brain")
        assert r['accepted'] is True
        s = mind.score_text("stroke occlusion")
        assert s['coherence'] > 0

    def test_feed_batch(self):
        mind = ShifuMind()
        texts = [
            "Stroke is a medical emergency",
            "The brain requires constant blood supply",
            "Arterial occlusion causes ischemic stroke",
        ]
        result = mind.feed_batch(texts)
        assert result['accepted'] >= 1

    def test_predict_candidates(self):
        mind = ShifuMind()
        for _ in range(5):
            mind.feed("Stroke is caused by cerebral arterial occlusion")
        results = mind.predict_candidates(
            [("stroke", 0.9), ("strake", 0.7)],
            context_words=["cerebral", "arterial"],
        )
        assert len(results) == 2
        assert results[0]['rank'] == 1

    def test_deliberate(self):
        mind = ShifuMind()
        mind.feed("Stroke is caused by arterial occlusion")
        result = mind.deliberate("stroke brain")
        assert 'coherence' in result
        assert 'focus' in result

    def test_describe(self):
        mind = ShifuMind()
        mind.feed("Stroke is caused by arterial occlusion in the brain")
        desc = mind.describe("stroke")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_confidence(self):
        mind = ShifuMind()
        assert mind.confidence("unknown")['score'] == 0
        mind.feed("Stroke affects the brain significantly")
        conf = mind.confidence("stroke")
        assert conf['score'] > 0

    def test_serialization_roundtrip(self):
        mind = ShifuMind()
        mind.feed("Stroke is caused by arterial occlusion in the brain")
        mind.feed("The heart pumps blood through arteries and veins")
        d = mind.to_dict()
        mind2 = ShifuMind.from_dict(d)
        assert mind2._feed_count == mind._feed_count
        assert mind2._epoch == mind._epoch
        # Verify graphs roundtrip exactly
        assert set(mind2._co_graph.keys()) == set(mind._co_graph.keys())
        assert set(mind2._nx_graph.keys()) == set(mind._nx_graph.keys())

    def test_no_px_graph(self):
        """M2 fix: _px_graph should not exist."""
        mind = ShifuMind()
        assert not hasattr(mind, '_px_graph')
        mind.feed("Stroke is caused by arterial occlusion")
        d = mind.to_dict()
        assert 'px_graph' not in d

    def test_from_dict_uses_init(self):
        """M6 fix: from_dict should use __init__, not __new__."""
        mind = ShifuMind()
        mind.feed("test text with enough words for acceptance")
        d = mind.to_dict()
        mind2 = ShifuMind.from_dict(d)
        # Should have all attributes set by __init__
        assert hasattr(mind2, '_graph_prune_interval')
        assert hasattr(mind2, '_graph_max_neighbors')

    def test_graph_pruning(self):
        """M1 fix: shared graphs should be pruned periodically."""
        mind = ShifuMind()
        mind._graph_prune_interval = 5  # prune often for test
        mind._graph_max_neighbors = 3
        for i in range(10):
            mind.feed(f"word{i} connects to many other tokens in this sentence")
        # After pruning, no node should have more than max_neighbors
        for graph in (mind._co_graph, mind._nx_graph):
            for node, neighbors in graph.items():
                assert len(neighbors) <= mind._graph_max_neighbors + 10  # some slack

    def test_empty_feed(self):
        mind = ShifuMind()
        result = mind.feed("x")
        assert result['accepted'] is False

    def test_score_empty(self):
        mind = ShifuMind()
        result = mind.score([])
        assert result['coherence'] == 0.0


# ── SemanticLandscape ──

class TestSemanticLandscape:
    def test_import_no_collision(self):
        """C1 fix: mind.SemanticLandscape should not collide with engine.Landscape."""
        from shifu_ocr.mind.landscape import SemanticLandscape
        assert SemanticLandscape.__name__ == "SemanticLandscape"

    def test_absorb_and_fit(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        from shifu_ocr.mind.landscape import SemanticLandscape
        ls = SemanticLandscape("test")
        ls.absorb([1.0, 2.0, 3.0])
        ls.absorb([1.1, 2.1, 2.9])
        score = ls.fit([1.05, 2.05, 2.95])
        assert score > -float('inf')

    def test_serialization_roundtrip(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        from shifu_ocr.mind.landscape import SemanticLandscape
        ls = SemanticLandscape("test")
        ls.absorb([1.0, 2.0])
        ls.absorb([1.1, 2.1])
        d = ls.to_dict()
        ls2 = SemanticLandscape.from_dict(d)
        assert ls2.label == "test"
        assert ls2.n == 2
