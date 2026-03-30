from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)


def fonts():
    try:
        return (
            ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 42),
            ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26),
        )
    except Exception:
        return (ImageFont.load_default(), ImageFont.load_default())


def make_shapes():
    fb, fm = fonts()
    img = Image.new('RGB', (960, 640), (245, 248, 252))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((60, 60, 900, 580), radius=24, outline=(30, 60, 100), width=4, fill=(255, 255, 255))
    d.rectangle((110, 140, 360, 420), fill=(70, 130, 200), outline=(20, 50, 100), width=3)
    d.ellipse((460, 160, 700, 400), fill=(230, 90, 90), outline=(140, 30, 30), width=4)
    d.polygon([(760, 420), (850, 200), (880, 470)], fill=(80, 180, 100), outline=(20, 100, 40))
    d.text((90, 80), 'Sample 1: Shapes', font=fb, fill=(20, 30, 50))
    d.text((110, 450), 'blue rectangle', font=fm, fill=(20, 30, 50))
    d.text((470, 420), 'red circle', font=fm, fill=(20, 30, 50))
    d.text((700, 500), 'green triangle', font=fm, fill=(20, 30, 50))
    path = ASSETS / 'sample_shapes.png'
    img.save(path)
    return path


def make_chart():
    fb, fm = fonts()
    img = Image.new('RGB', (960, 640), (250, 250, 245))
    d = ImageDraw.Draw(img)
    d.text((80, 60), 'Sample 2: Sales by Quarter', font=fb, fill=(30, 40, 50))
    d.line((120, 500, 860, 500), fill=(0, 0, 0), width=3)
    d.line((120, 160, 120, 500), fill=(0, 0, 0), width=3)
    bars = [('Q1', 120, (90, 140, 220)), ('Q2', 220, (120, 190, 120)), ('Q3', 320, (235, 140, 80)), ('Q4', 260, (200, 90, 140))]
    x = 180
    for label, h, color in bars:
        d.rectangle((x, 500 - h, x + 90, 500), fill=color, outline=(40, 40, 40), width=2)
        d.text((x + 18, 515), label, font=fm, fill=(20, 20, 20))
        d.text((x + 18, 500 - h - 35), str(h), font=fm, fill=(20, 20, 20))
        x += 150
    d.text((120, 560), 'Units sold', font=fm, fill=(50, 50, 50))
    path = ASSETS / 'sample_chart.png'
    img.save(path)
    return path


def make_document():
    fb, fm = fonts()
    img = Image.new('RGB', (960, 640), 'white')
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((80, 50, 880, 590), radius=14, outline=(80, 80, 80), width=2)
    d.text((120, 90), 'Sample 3: Receipt', font=fb, fill=(20, 20, 20))
    lines = [
        'Store: OpenClaw Mart',
        'Date: 2026-03-30',
        'Milk            2 x 180',
        'Bread           1 x 240',
        'Apples          3 x 120',
        '------------------------',
        'Subtotal            960',
        'Tax                  96',
        'Total              1056',
    ]
    y = 170
    for line in lines:
        d.text((140, y), line, font=fm, fill=(30, 30, 30))
        y += 42
    path = ASSETS / 'sample_receipt.png'
    img.save(path)
    return path


def main():
    paths = [make_shapes(), make_chart(), make_document()]
    for p in paths:
        print(p)


if __name__ == '__main__':
    main()
