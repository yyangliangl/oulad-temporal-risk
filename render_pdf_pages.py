from pathlib import Path
import sys
import pypdfium2 as pdfium

root = Path(__file__).resolve().parents[1]
pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "outputs" / "ICETM2026_IoT_Virtual_Sensing_Draft.pdf"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "outputs" / "rendered"
out.mkdir(exist_ok=True)
pdf = pdfium.PdfDocument(pdf_path)
print(f"pages={len(pdf)}")
for i, page in enumerate(pdf):
    bitmap = page.render(scale=1.7)
    image = bitmap.to_pil()
    image.save(out / f"page-{i+1}.png")
