// SHIFU v2 — Autonomous Teaching Loop
// Connects an LLM (Claude, Gemma, any OpenAI-compatible API) to the Teacher.
// The LLM reads Shifu's pressure map and generates exactly what it needs.
// Senior doctor (LLM) teaching a junior (Shifu). Unlimited conversation.
//
// Usage:
//   const tutor = new Tutor(engine, teacher, { provider: "anthropic", apiKey: "..." });
//   await tutor.session("medical", 10); // 10 rounds of autonomous teaching
//
// Or with Gemma/Ollama (local, free, unlimited):
//   const tutor = new Tutor(engine, teacher, { provider: "ollama", model: "gemma3:4b" });

class Tutor {
  constructor(engine, teacher, options = {}) {
    this.engine = engine;
    this.teacher = teacher;
    this.provider = options.provider || "anthropic";  // "anthropic", "openai", "ollama"
    this.apiKey = options.apiKey || process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || null;
    this.model = options.model || (this.provider === "anthropic" ? "claude-sonnet-4-20250514" : this.provider === "ollama" ? "gemma3:4b" : "gpt-4o-mini");
    this.baseUrl = options.baseUrl || (this.provider === "ollama" ? "http://localhost:11434" : this.provider === "anthropic" ? "https://api.anthropic.com" : "https://api.openai.com");
    this.domain = options.domain || "medical";
    this.rounds = [];
    this.totalTokensUsed = 0;
  }

  // ─── Ask the LLM ───────────────────────────────────────────────
  async ask(prompt, system = null) {
    if (this.provider === "anthropic") return this._askAnthropic(prompt, system);
    if (this.provider === "ollama") return this._askOllama(prompt, system);
    return this._askOpenAI(prompt, system);
  }

  async _askAnthropic(prompt, system) {
    const res = await fetch(`${this.baseUrl}/v1/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": this.apiKey, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: this.model, max_tokens: 1024, system: system || "", messages: [{ role: "user", content: prompt }] }),
    });
    const data = await res.json();
    return data.content?.[0]?.text || "";
  }

  async _askOllama(prompt, system) {
    const res = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, prompt: system ? `${system}\n\n${prompt}` : prompt, stream: false }),
    });
    const data = await res.json();
    return data.response || "";
  }

  async _askOpenAI(prompt, system) {
    const msgs = [];
    if (system) msgs.push({ role: "system", content: system });
    msgs.push({ role: "user", content: prompt });
    const res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${this.apiKey}` },
      body: JSON.stringify({ model: this.model, messages: msgs, max_tokens: 1024 }),
    });
    const data = await res.json();
    return data.choices?.[0]?.message?.content || "";
  }

  // ─── The System Prompt: who the LLM is ─────────────────────────
  systemPrompt() {
    return `You are a senior ${this.domain} specialist teaching a junior system called Shifu.
Shifu learns language by exposure to sentences. It has no pre-existing knowledge — only what you feed it.
Your job: generate clear, factual, ${this.domain} sentences that teach Shifu vocabulary and relationships.

Rules:
- Write 5-10 simple, factual sentences per response
- Each sentence should be self-contained (Shifu processes one sentence at a time)
- Use consistent terminology (don't switch between synonyms randomly)
- When told to focus on specific words, use those words in varied contexts
- When told to create contrastive pairs, write a sentence AND its meaningful reversal
- No questions, no meta-commentary — just sentences Shifu can learn from
- Write at a clinical/professional level, not textbook definitions`;
  }

  // ─── One teaching round ────────────────────────────────────────
  async round() {
    const diag = this.teacher.diagnose();
    const plan = this.teacher.plan();
    const roundNum = this.rounds.length + 1;

    // Build the prompt from the plan
    let prompt = `Round ${roundNum}. Shifu's current state:\n`;
    prompt += `- Vocabulary: ${diag.vocab} words\n`;
    prompt += `- Depth distribution: ${JSON.stringify(diag.depths)}\n`;
    prompt += `- Needs: ${diag.needs.join(" | ")}\n\n`;

    if (plan.steps.length > 0) {
      const step = plan.steps[0]; // highest priority
      prompt += `Priority task: ${step.instruction}\n\n`;
      if (step.targets?.length) {
        prompt += `Target words to use: ${step.targets.join(", ")}\n\n`;
      }
    }

    prompt += `Generate 5-10 ${this.domain} sentences for Shifu to learn.`;

    // Ask the LLM
    const text = await this.ask(prompt, this.systemPrompt());
    if (!text.trim()) return { round: roundNum, error: "Empty response from LLM" };

    // Feed to Shifu via Teacher (lesson = feed + decay + report)
    const lesson = this.teacher.lesson(text, this.domain);

    // Pick a sentence for contrastive drill
    const sents = text.split(/[.!?\n]+/).map(s => s.trim()).filter(s => s.length > 10);
    let drill = null;
    if (sents.length > 0) {
      drill = this.teacher.drill(sents[0]);
    }

    // Check for bridges that might be contamination
    const bridges = this.engine.bridges(3);
    let corrections = [];
    if (bridges.length > 0) {
      // Ask LLM if these bridges are legitimate
      const bridgePrompt = `In ${this.domain}, are these words legitimately cross-domain bridges, or contamination? Answer ONLY "keep" or "remove" for each:\n${bridges.map(b => `- "${b.word}" (closure: ${b.closure.toFixed(2)}, connects disconnected word groups)`).join("\n")}`;
      const bridgeAnswer = await this.ask(bridgePrompt, this.systemPrompt());
      // Parse simple keep/remove responses
      for (const bridge of bridges) {
        if (bridgeAnswer.toLowerCase().includes(`${bridge.word}`) && bridgeAnswer.toLowerCase().includes("remove")) {
          this.engine.forget(bridge.word);
          corrections.push({ action: "forgot", word: bridge.word, reason: "LLM flagged as contamination" });
        }
      }
    }

    const result = {
      round: roundNum,
      llmGenerated: text.slice(0, 200) + (text.length > 200 ? "..." : ""),
      lesson,
      drill: drill ? { sentence: drill.sentence, asymmetry: drill.asymmetry, learned: drill.learned } : null,
      corrections,
      afterState: this.engine.stats(),
    };
    this.rounds.push(result);
    return result;
  }

  // ─── Full teaching session ─────────────────────────────────────
  async session(domain, numRounds = 10, options = {}) {
    if (domain) this.domain = domain;
    const onRound = options.onRound || null; // callback(result) per round
    const delayMs = options.delayMs || 1000; // pause between rounds (rate limiting)

    console.log(`\n═══ SHIFU TEACHING SESSION: ${this.domain} ═══`);
    console.log(`Provider: ${this.provider}, Model: ${this.model}`);
    console.log(`Rounds: ${numRounds}\n`);

    const beforeState = this.engine.stats();

    for (let i = 0; i < numRounds; i++) {
      try {
        const result = await this.round();
        if (result.error) {
          console.log(`  Round ${result.round}: ERROR — ${result.error}`);
          continue;
        }
        console.log(`  Round ${result.round}: +${result.lesson.newWords} words, vocab ${result.afterState.vocab}, drill asymmetry: ${result.drill?.asymmetry?.toFixed(3) || "n/a"}`);
        if (result.corrections.length) console.log(`    Corrections: ${result.corrections.map(c => `${c.action} "${c.word}"`).join(", ")}`);
        if (onRound) onRound(result);
      } catch (e) {
        console.log(`  Round ${i + 1}: EXCEPTION — ${e.message}`);
      }
      if (i < numRounds - 1 && delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
    }

    const afterState = this.engine.stats();
    const summary = {
      domain: this.domain,
      rounds: this.rounds.length,
      before: beforeState,
      after: afterState,
      vocabGrowth: afterState.vocab - beforeState.vocab,
      depthProgress: afterState.depths,
      finalNeeds: this.teacher.diagnose().needs,
    };
    console.log(`\n═══ SESSION COMPLETE ═══`);
    console.log(`  Vocab: ${beforeState.vocab} → ${afterState.vocab} (+${summary.vocabGrowth})`);
    console.log(`  Depths: ${JSON.stringify(afterState.depths)}`);
    console.log(`  Remaining needs: ${summary.finalNeeds[0]?.slice(0, 80)}...`);
    return summary;
  }

  // ─── Ask the LLM to evaluate Shifu's understanding ─────────────
  async evaluate() {
    const diag = this.teacher.diagnose();

    // Get Shifu's strongest words
    const deep = Object.entries(this.engine.nodes)
      .filter(([w]) => this.engine.depth(w).level === "deep" || this.engine.depth(w).level === "structured")
      .map(([w]) => w).slice(0, 10);

    // Ask LLM to generate test sentences
    const prompt = `Shifu knows these ${this.domain} words well: ${deep.join(", ")}.
Generate 5 test sentences using these words — some correct, some with wrong word order or wrong associations.
Format: one sentence per line, prefix with CORRECT: or WRONG:`;

    const text = await this.ask(prompt, this.systemPrompt());
    const lines = text.split("\n").filter(l => l.trim());

    const results = [];
    for (const line of lines) {
      const isCorrect = line.toLowerCase().startsWith("correct:");
      const isWrong = line.toLowerCase().startsWith("wrong:");
      if (!isCorrect && !isWrong) continue;
      const sentence = line.replace(/^(correct|wrong):\s*/i, "").trim();
      const score = this.engine.scoreSentence(sentence);
      results.push({
        sentence,
        expected: isCorrect ? "correct" : "wrong",
        coherence: score.coherence,
        shifuSays: score.coherence > 0.4 ? "correct" : "wrong",
        match: (isCorrect && score.coherence > 0.4) || (isWrong && score.coherence <= 0.4),
      });
    }

    const accuracy = results.length ? results.filter(r => r.match).length / results.length : 0;
    return { results, accuracy, tested: results.length };
  }
}

module.exports = { Tutor };
