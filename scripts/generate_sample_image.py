from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    out = Path(__file__).resolve().parents[1] / 'assets' / 'sample_shapes.png'
    out.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new('RGB', (960, 640), color=(245, 248, 252))
    draw = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 42)
        font_mid = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    except Exception:
        font_big = ImageFont.load_default()
        font_mid = ImageFont.load_default()

    draw.rounded_rectangle((60, 60, 900, 580), radius=24, outline=(30, 60, 100), width=4, fill=(255, 255, 255))
    draw.rectangle((110, 140, 360, 420), fill=(70, 130, 200), outline=(20, 50, 100), width=3)
    draw.ellipse((460, 160, 700, 400), fill=(230, 90, 90), outline=(140, 30, 30), width=4)
    draw.polygon([(760, 420), (850, 200), (880, 470)], fill=(80, 180, 100), outline=(20, 100, 40))
    draw.text((90, 80), 'Sample Shapes', font=font_big, fill=(20, 30, 50))
    draw.text((110, 450), 'blue rectangle', font=font_mid, fill=(20, 30, 50))
    draw.text((470, 420), 'red circle', font=font_mid, fill=(20, 30, 50))
    draw.text((700, 500), 'green triangle', font=font_mid, fill=(20, 30, 50))
    img.save(out)
    print(out)


if __name__ == '__main__':
    main()
