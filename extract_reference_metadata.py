from pathlib import Path
from pypdf import PdfReader

src = Path(r"D:\Codex\References")
out = Path(__file__).resolve().parents[1] / "outputs" / "reference_metadata.txt"
blocks = []
for pdf in sorted(src.glob("*.pdf")):
    try:
        reader = PdfReader(pdf)
        meta = reader.metadata or {}
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:2])
        blocks.append(
            f"===== {pdf.name} =====\n"
            f"TITLE_META: {meta.get('/Title', '')}\n"
            f"AUTHOR_META: {meta.get('/Author', '')}\n"
            f"PAGES: {len(reader.pages)}\n{text[:7000]}\n"
        )
    except Exception as exc:
        blocks.append(f"===== {pdf.name} =====\nERROR: {exc}\n")
out.write_text("\n".join(blocks), encoding="utf-8")
print(out)
