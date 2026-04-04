#!/usr/bin/env node
// Usage:
//   ANTHROPIC_API_KEY=sk-... node tutor_run.js medical 20
//   Or with local Gemma (free, unlimited):
//   node tutor_run.js medical 50 ollama gemma3:4b

const { ShifuEmbryo } = require("./core/engine");
const { Teacher } = require("./core/teacher");
const { Tutor } = require("./core/tutor");
const fs = require("fs");

const domain = process.argv[2] || "medical";
const rounds = parseInt(process.argv[3] || "10");
const provider = process.argv[4] || "anthropic";
const model = process.argv[5] || undefined;
const savePath = `./data/shifu_${domain}.json`;

async function main() {
  // Load or create engine
  let engine;
  if (fs.existsSync(savePath)) {
    engine = ShifuEmbryo.deserialize(fs.readFileSync(savePath, "utf8"));
    console.log(`Loaded ${domain} engine: ${engine.stats().vocab} words`);
  } else {
    engine = new ShifuEmbryo();
    console.log(`Fresh ${domain} engine`);
  }

  const teacher = new Teacher(engine);
  const tutor = new Tutor(engine, teacher, { provider, model });

  // Run session
  const summary = await tutor.session(domain, rounds, {
    delayMs: provider === "ollama" ? 100 : 1500, // local = fast, API = rate limit
  });

  // Evaluate
  console.log("\n═══ EVALUATION ═══");
  const eval_ = await tutor.evaluate();
  console.log(`  Tested: ${eval_.tested} sentences`);
  console.log(`  Accuracy: ${(eval_.accuracy * 100).toFixed(0)}%`);
  for (const r of eval_.results) {
    console.log(`  ${r.match ? "✓" : "✗"} [${r.expected}] coherence ${r.coherence.toFixed(3)}: ${r.sentence.slice(0, 60)}`);
  }

  // Save engine state
  if (!fs.existsSync("./data")) fs.mkdirSync("./data");
  fs.writeFileSync(savePath, engine.serialize());
  console.log(`\nSaved to ${savePath}`);
}

main().catch(e => { console.error(e); process.exit(1); });
