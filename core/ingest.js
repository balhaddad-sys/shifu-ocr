// Shifu Real-Data Ingestion Pipeline
// Processes actual hospital documents: ward sheets, lab reports, PDFs.
//
// Supports:
//   - PDF files (via pdf-parse or external tools)
//   - Image files (PNG, JPG, TIFF) → Python OCR pipeline
//   - CSV/TSV files (structured ward census data)
//   - Plain text files (clinical notes)
//
// Each input becomes a standardized document that flows through the
// correction + feedback pipeline.

const path = require('path');
const fs = require('fs');

class DocumentIngestor {
  constructor(pipeline, options = {}) {
    this.pipeline = pipeline; // ShifuPipeline instance
    this.outputDir = options.outputDir || path.join(__dirname, '..', '.ingested');
    this.persistArtifacts = options.persistArtifacts || false;
    if (this.persistArtifacts) this._ensureDir(this.outputDir);
  }

  /**
   * Ingest a file. Auto-detects format from extension.
   * Returns a standardized document with raw + corrected text.
   */
  async ingestFile(filePath, options = {}) {
    if (!fs.existsSync(filePath)) {
      return { error: `File not found: ${filePath}`, source: filePath };
    }

    let ext = path.extname(filePath).toLowerCase();
    const basename = path.basename(filePath);

    // When extension is missing or unknown, detect type from file magic bytes
    if (!ext || !['.csv','.tsv','.txt','.png','.jpg','.jpeg','.tiff','.tif','.bmp','.pdf'].includes(ext)) {
      try {
        const head = Buffer.alloc(8);
        let fd;
        try {
          fd = fs.openSync(filePath, 'r');
          fs.readSync(fd, head, 0, 8, 0);
        } finally {
          if (fd !== undefined) fs.closeSync(fd);
        }
        if (head[0] === 0x25 && head[1] === 0x50 && head[2] === 0x44 && head[3] === 0x46) {
          ext = '.pdf';   // %PDF
        } else if (head[0] === 0x89 && head[1] === 0x50 && head[2] === 0x4E && head[3] === 0x47) {
          ext = '.png';   // PNG
        } else if (head[0] === 0xFF && head[1] === 0xD8 && head[2] === 0xFF) {
          ext = '.jpg';   // JPEG
        } else if (head[0] === 0x42 && head[1] === 0x4D) {
          ext = '.bmp';   // BMP
        } else if (
          ((head[0] === 0x49 && head[1] === 0x49 && head[2] === 0x2A && head[3] === 0x00) ||
           (head[0] === 0x4D && head[1] === 0x4D && head[2] === 0x00 && head[3] === 0x2A))
        ) {
          ext = '.tiff';  // TIFF (II+0x002A little-endian or MM+0x002A big-endian)
        }
        // Otherwise stays as-is and falls through to text handler
      } catch {}
    }

    let doc;
    try {
    switch (ext) {
      case '.csv':
      case '.tsv':
        doc = await this._ingestCSV(filePath, ext === '.tsv' ? '\t' : ',');
        break;
      case '.txt':
        doc = await this._ingestText(filePath, options);
        break;
      case '.png':
      case '.jpg':
      case '.jpeg':
      case '.tiff':
      case '.tif':
      case '.bmp':
        doc = await this._ingestImage(filePath);
        break;
      case '.pdf':
        doc = await this._ingestPDF(filePath, options);
        break;
      default:
        // Try as plain text
        doc = await this._ingestText(filePath, options);
    }
    } catch (err) {
      return { error: `Failed to process ${filePath}: ${err.message}`, source: filePath, overallDecision: 'verify' };
    }

    doc.source = filePath;
    doc.filename = basename;
    doc.ingestedAt = new Date().toISOString();

    // Only persist artifacts to disk if explicitly enabled
    if (this.persistArtifacts) {
      const ts = Date.now();
      const outputPath = path.join(this.outputDir, `${ts}_${basename}.json`);
      fs.writeFileSync(outputPath, JSON.stringify(doc, null, 2));
    }

    return doc;
  }

  /**
   * Ingest a directory of files.
   */
  async ingestDirectory(dirPath, options = {}) {
    const extensions = options.extensions || ['.csv', '.tsv', '.txt', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.pdf'];
    const files = fs.readdirSync(dirPath)
      .filter(f => {
        const ext = path.extname(f).toLowerCase();
        // Include files with matching extensions OR extensionless files (magic byte detection handles these)
        return extensions.includes(ext) || ext === '';
      })
      .filter(f => fs.statSync(path.join(dirPath, f)).isFile())
      .map(f => path.join(dirPath, f));

    const results = [];
    for (const file of files) {
      const doc = await this.ingestFile(file, options);
      results.push(doc);
    }
    return { files: results.length, documents: results };
  }

  /**
   * Ingest raw text directly (e.g., from clipboard or API).
   */
  ingestRawText(text, options = {}) {
    const format = options.format || 'line';
    if (format === 'table' && options.columns) {
      return this._processTableText(text, options.columns, options.delimiter || ',');
    }
    return this._processLineText(text, { correct: options.correct !== false, sourceMode: 'raw_text' });
  }

  /**
   * Ingest a ward census table (the core use case).
   * Takes an array of row objects: [{ Patient: '...', Diagnosis: '...', ... }]
   */
  ingestWardCensus(rows) {
    const results = [];
    for (const row of rows) {
      const corrected = this.pipeline.processTableRow(row);
      results.push({
        raw: row,
        corrected: corrected.corrected,
        safetyFlags: corrected.safetyFlags,
        decision: corrected.decision,
      });
    }

    const accepted = results.filter(r => r.decision === 'accept').length;
    const verify = results.filter(r => r.decision === 'verify').length;
    const rejected = results.filter(r => r.decision === 'reject').length;

    return {
      type: 'ward_census',
      rows: results,
      summary: {
        total: results.length,
        accepted,
        needsVerification: verify,
        rejected,
        accuracy: results.length > 0 ? accepted / results.length : 0,
      },
    };
  }

  // ─── Internal ─────────────────────────────────────────────────

  async _ingestImage(filePath) {
    const result = await this.pipeline.processImage(filePath, { format: 'line' });
    return {
      type: 'image',
      format: 'line',
      lines: result.lines || [],
      overallDecision: result.overallDecision,
      raw: result.raw,
    };
  }

  async _ingestCSV(filePath, delimiter) {
    const content = fs.readFileSync(filePath, 'utf-8');
    const records = this._parseCSVRecords(content, delimiter);
    if (records.length === 0) return { type: 'csv', rows: [], columns: [] };

    const columns = records[0];
    const rows = [];

    for (let i = 1; i < records.length; i++) {
      const cells = records[i];
      const row = {};
      columns.forEach((col, j) => { row[col] = cells[j] || ''; });

      const corrected = this.pipeline.processTableRow(row);
      rows.push({
        raw: row,
        corrected: corrected.corrected,
        safetyFlags: corrected.safetyFlags,
        decision: corrected.decision,
      });
    }

    return { type: 'csv', columns, rows };
  }

  /**
   * Parse a CSV line respecting quoted fields.
   * Handles: `Bader,"stroke, ischemic",Dr. Saleh`
   */
  _parseCSVLine(line, delimiter) {
    const fields = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
          current += '"'; // escaped quote
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === delimiter && !inQuotes) {
        fields.push(current.trim());
        current = '';
      } else {
        current += ch;
      }
    }
    fields.push(current.trim());
    return fields;
  }

  /**
   * Parse full CSV content into records, handling multiline quoted fields.
   * `Bader,"line one\nline two",Dr. Saleh` becomes one record with 3 fields.
   */
  _parseCSVRecords(content, delimiter) {
    const records = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < content.length; i++) {
      const ch = content[i];
      if (ch === '"') {
        if (inQuotes && i + 1 < content.length && content[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes; // toggle quote mode, don't add delimiter to output
        }
      } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
        // End of record
        if (ch === '\r' && i + 1 < content.length && content[i + 1] === '\n') i++;
        if (current.trim()) {
          records.push(this._parseCSVLine(current, delimiter));
        }
        current = '';
      } else {
        current += ch;
      }
    }
    if (current.trim()) {
      records.push(this._parseCSVLine(current, delimiter));
    }
    return records;
  }

  async _ingestText(filePath, options = {}) {
    const content = fs.readFileSync(filePath, 'utf-8');
    return this._processLineText(content, {
      correct: options.correctNativeText === true,
      sourceMode: 'native_text',
    });
  }

  _processLineText(content, options = {}) {
    const correct = options.correct !== false;
    const rawLines = content.split('\n').filter(l => l.trim());
    const processed = rawLines.map(line => this.pipeline.processText(line, { correct }));
    return {
      type: 'text',
      sourceMode: options.sourceMode || (correct ? 'ocr_text' : 'native_text'),
      rawText: content,
      rawLines,
      lines: processed,
      overallDecision: processed.some(r => r.hasDangers) ? 'reject'
        : processed.some(r => r.decision === 'verify' || r.decision === 'reject' || r.hasWarnings) ? 'verify' : 'accept',
    };
  }

  _processTableText(content, columns, delimiter) {
    const records = this._parseCSVRecords(content, delimiter);
    const rows = records.map(cells => {
      const row = {};
      columns.forEach((col, i) => { row[col] = cells[i] || ''; });
      return row;
    });
    return this.ingestWardCensus(rows);
  }

  async _ingestPDF(filePath, options = {}) {
    // Try to extract text from PDF using external tools
    const { spawn } = require('child_process');

    // First try: PyMuPDF (best quality), then pdfplumber, then PyPDF2
    const text = await new Promise((resolve) => {
      const script = `
import sys
text_found = False

# Try PyMuPDF first (best word spacing)
try:
    import fitz
    doc = fitz.open(sys.argv[1])
    for page in doc:
        text = page.get_text()
        if text.strip():
            sys.stdout.write(text)
            sys.stdout.flush()
            text_found = True
    doc.close()
except ImportError:
    pass
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)

if not text_found:
    # Fallback to pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(sys.argv[1]) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ''
                if text:
                    sys.stdout.write(text + '\\n')
                    sys.stdout.flush()
                    text_found = True
    except ImportError:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)

if not text_found:
    # Fallback to PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(sys.argv[1])
        for page in reader.pages:
            text = page.extract_text() or ''
            if text:
                sys.stdout.write(text + '\\n')
                sys.stdout.flush()
    except ImportError:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
`;
      let proc;
      try {
        proc = spawn('python', ['-c', script, filePath], { timeout: 600000 });
      } catch {
        resolve('');
        return;
      }

      let stdout = '';
      proc.stdout.on('data', d => { stdout += d.toString(); });
      proc.on('close', () => resolve(stdout.trim()));
      proc.on('error', () => resolve(''));
    });

    if (text) {
      const pageCount = (text.match(/\f/g) || []).length + 1;
      const result = this._processLineText(text, {
        correct: options.correctNativeText === true,
        sourceMode: 'native_pdf_text',
      });
      result.type = 'pdf';
      result.extractionMethod = 'native_text';
      result.extractionDetail = 'Text extracted directly from PDF using PyMuPDF/pdfplumber/PyPDF2';
      result.pageCount = pageCount;
      result.charCount = text.length;
      return result;
    }

    // Fallback: convert PDF pages to images via PyMuPDF, then OCR each page
    try {
      const tmpDir = path.join(require('os').tmpdir(), 'shifu_pdf_' + Date.now());
      const convertScript = `
import sys, os, fitz
doc = fitz.open(sys.argv[1])
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=200)
    out_path = os.path.join(out_dir, f"page_{i:04d}.png")
    pix.save(out_path)
    print(out_path)
doc.close()
`;
      const pages = await new Promise((resolve) => {
        let proc;
        try {
          proc = spawn('python', ['-c', convertScript, filePath, tmpDir], { timeout: 120000 });
        } catch {
          resolve([]);
          return;
        }
        let stdout = '';
        proc.stdout.on('data', d => { stdout += d.toString(); });
        proc.on('close', () => resolve(stdout.trim().split('\n').filter(Boolean)));
        proc.on('error', () => resolve([]));
      });

      if (pages.length > 0) {
        const allLines = [];
        let worstDecision = 'accept';
        for (const pagePath of pages) {
          const result = await this.pipeline.processImage(pagePath, { format: 'line' });
          if (result && result.lines) allLines.push(...result.lines);
          if (result && result.overallDecision === 'verify') worstDecision = 'verify';
        }
        // Clean up temp images
        try { pages.forEach(p => fs.unlinkSync(p)); fs.rmdirSync(tmpDir); } catch {}

        if (allLines.length > 0) {
          return {
            type: 'pdf',
            format: 'line',
            lines: allLines,
            overallDecision: worstDecision,
            sourceMode: 'ocr_fallback',
          };
        }
      }
    } catch (e) {
      // OCR fallback failed, continue to error
    }

    return { type: 'pdf', lines: [], error: 'Could not extract text from PDF (no text layer found and OCR fallback failed).', overallDecision: 'verify' };
  }

  _ensureDir(dir) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  }
}

module.exports = { DocumentIngestor };
