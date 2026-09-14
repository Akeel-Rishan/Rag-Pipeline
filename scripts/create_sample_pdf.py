"""Create the project's tiny, synthetic PDF fixture without extra libraries."""

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def create_sample_pdf(path: Path) -> None:
    """Write text / blank / text pages. Refuse to overwrite an existing file."""
    with PdfWriter() as writer:
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        for text in ("Document loading preserves source metadata.", "",
                     "This is physical page three."):
            page = writer.add_blank_page(width=612, height=792)
            if text:
                page[NameObject("/Resources")] = DictionaryObject({
                    NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
                })
                # PDF drawing instructions: begin text, select font, position,
                # draw a fixed ASCII string, and end text. Not a general writer.
                content = DecodedStreamObject()
                content.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
                page[NameObject("/Contents")] = content
        with path.open("xb") as stream:
            writer.write(stream)


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "data/sample/example.pdf"
    create_sample_pdf(destination)
    print(f"Created {destination}")
