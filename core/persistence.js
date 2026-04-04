// Shifu Persistence Layer
// Auto-saves and loads the learning state between sessions.
// The engine gets smarter every session — this is what makes that permanent.

const fs = require('fs');
const path = require('path');

const DEFAULT_STATE_DIR = path.join(__dirname, '..', '.state');
const CORE_STATE_FILE = 'core_engine.json';
const LEARNING_STATE_FILE = 'learning_engine.json';
const META_FILE = 'meta.json';

class ShifuPersistence {
  constructor(stateDir = DEFAULT_STATE_DIR) {
    this.stateDir = stateDir;
    this._ensureDir();
  }

  _ensureDir() {
    if (!fs.existsSync(this.stateDir)) fs.mkdirSync(this.stateDir, { recursive: true });
  }

  /**
   * Save the full Shifu system state.
   * Called after learning from corrections, or periodically.
   */
  save(shifu) {
    if (this._isSaving) return;
    this._isSaving = true;
    const state = shifu.serialize();

    try {
      this._ensureDir();
      
      const writeAtomic = (fileName, content) => {
        const finalPath = path.join(this.stateDir, fileName);
        const tmpPath = finalPath + '.tmp';
        fs.writeFileSync(tmpPath, content);
        fs.renameSync(tmpPath, finalPath);
      };

      writeAtomic(
        CORE_STATE_FILE,
        typeof state.core === 'string' ? state.core : JSON.stringify(state.core)
      );

      writeAtomic(
        LEARNING_STATE_FILE,
        JSON.stringify(state.learning)
      );

      writeAtomic(
        META_FILE,
        JSON.stringify({
          version: state.version,
          savedAt: state.savedAt,
          coreSize: fs.statSync(path.join(this.stateDir, CORE_STATE_FILE)).size,
          learningSize: fs.statSync(path.join(this.stateDir, LEARNING_STATE_FILE)).size,
        }, null, 2)
      );
    } catch (err) {
      console.warn(`Failed to save state (rollback active): ${err.message}`);
    } finally {
      this._isSaving = false;
    }

    return state.savedAt;
  }

  /**
   * Load saved state. Returns { savedCoreState, savedLearningState } or null.
   */
  load() {
    const corePath = path.join(this.stateDir, CORE_STATE_FILE);
    const learningPath = path.join(this.stateDir, LEARNING_STATE_FILE);

    if (!fs.existsSync(corePath) || !fs.existsSync(learningPath)) return null;

    // Parse each file independently — corrupt meta or learning should not
    // discard valid state from other files
    let savedCoreState = null;
    try {
      savedCoreState = fs.readFileSync(corePath, 'utf-8');
    } catch (e) {
      console.warn(`Failed to load core state: ${e.message}`);
      return null;
    }

    let savedLearningState = null;
    try {
      savedLearningState = JSON.parse(fs.readFileSync(learningPath, 'utf-8'));
    } catch (e) {
      // Core and learning must be consistent — if learning is corrupt,
      // discard both to avoid split-brain (core remembers, learning doesn't)
      console.warn(`Corrupt learning_engine.json, starting fresh: ${e.message}`);
      return null;
    }

    let meta = {};
    try {
      if (fs.existsSync(path.join(this.stateDir, META_FILE))) {
        meta = JSON.parse(fs.readFileSync(path.join(this.stateDir, META_FILE), 'utf-8'));
      }
    } catch (e) {
      console.warn(`Corrupt meta.json, ignoring: ${e.message}`);
    }

    return { savedCoreState, savedLearningState, meta };
  }

  /**
   * Check if saved state exists.
   */
  hasSavedState() {
    return fs.existsSync(path.join(this.stateDir, CORE_STATE_FILE))
      && fs.existsSync(path.join(this.stateDir, LEARNING_STATE_FILE));
  }

  /**
   * Get metadata about saved state.
   */
  getMeta() {
    const metaPath = path.join(this.stateDir, META_FILE);
    if (!fs.existsSync(metaPath)) return null;
    try {
      return JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    } catch {
      return null;
    }
  }

  /**
   * Clear saved state (reset to fresh).
   */
  clear() {
    for (const file of [CORE_STATE_FILE, LEARNING_STATE_FILE, META_FILE]) {
      const p = path.join(this.stateDir, file);
      if (fs.existsSync(p)) fs.unlinkSync(p);
    }
  }
}

/**
 * Auto-saving wrapper. Wraps a shifu instance to auto-save after learn() calls.
 */
function withAutoSave(shifu, options = {}) {
  const persistence = new ShifuPersistence(options.stateDir);
  const saveInterval = options.saveInterval || 1; // Save after every correction by default
  let correctionsSinceLastSave = 0;

  const originalLearn = shifu.learn.bind(shifu);
  shifu.learn = function (ocrRow, confirmedRow) {
    const result = originalLearn(ocrRow, confirmedRow);
    // Only count successful learns toward the save interval
    if (result && result.accepted) {
      correctionsSinceLastSave++;
      if (correctionsSinceLastSave >= saveInterval) {
        persistence.save(shifu);
        correctionsSinceLastSave = 0;
      }
    }
    return result;
  };

  shifu.forceSave = () => persistence.save(shifu);
  shifu.persistence = persistence;

  return shifu;
}

module.exports = { ShifuPersistence, withAutoSave };
