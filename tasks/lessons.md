# Lessons Learned

## 2026-04-02: The Threading Trap

**Problem:** Brain process (Python) died after 2-5 minutes when using background threads for diastolic heartbeat loop.

**Root cause:** On Windows with piped stdin from Node.js, Python's GIL makes threading unreliable. Reader threads compete with the main loop for GIL access. `for line in sys.stdin` in a thread terminates unpredictably. `queue.Queue.get(timeout)` doesn't guarantee the reader thread gets scheduled. Even with zero background work, the thread pattern fails.

**Solution:** Remove ALL threads. Single-threaded `stdin.readline()` blocking in the main loop. Process command → heartbeat → respond → block again. The brain sleeps when idle, wakes when spoken to.

**Rule:** On Windows with piped stdin, NEVER use threads for stdin reading. Use blocking readline in the main loop. If you need background work, piggyback it on command processing — don't run it in a separate thread.

**Corollary:** The simplest solution is almost always the right one. Before adding threads, queues, locks, and condition variables — ask: "Can I just block on readline?"

## 2026-04-02: Don't Feed an Elephant to a Baby

**Problem:** 150MB brain process with neuroglia, TRN, 5-checkpoint relay, conviction, fatigue — all running on 26 words. Constant instability, timeouts, overengineering.

**Root cause:** Built an adult nervous system before the baby had teeth. The architecture should GROW with the data, not be born fully formed.

**Principles:**
1. Feed what the baby can swallow. Vocab < 50 = milk (1 sentence at a time). Vocab < 500 = puree (10 sentences). Vocab > 5000 = solid food (full batches).
2. Babies have MORE neurons, FEWER synapses. Build blocks first, refine later. Don't try to build connections before having enough neurons.
3. Distributed storage. Don't store the whole elephant in one dict. Visual info in one region, temporal in another, semantic in another. Each region holds a small piece = smaller load per region.

**Rule:** Before adding complexity, ask: "Does the baby have teeth for this?" If vocab < 100, it doesn't need neuroglia, TRN, conviction, fatigue, or 5-checkpoint relay. It needs to eat and grow.
