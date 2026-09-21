import base64
import io
import unittest

from PIL import Image

from media import MediaInputError, decode_data_url, prepare_media


def image_data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), (220, 30, 20)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class MediaTests(unittest.TestCase):
    def test_prepares_image_without_retaining_base64(self):
        prepared = prepare_media(
            [{"type": "image", "name": "red.png", "data_url": image_data_url()}],
            video_fps=1.0,
            max_video_frames=8,
        )
        self.assertEqual(prepared[0].content.size, (4, 3))
        self.assertNotIn("data_url", prepared[0].public_metadata())

    def test_rejects_mime_kind_mismatch(self):
        with self.assertRaises(MediaInputError):
            decode_data_url(image_data_url(), expected_kind="video")

    def test_rejects_multiple_videos_before_decoding(self):
        items = [
            {"type": "video", "data_url": "data:video/mp4;base64,AAAA"},
            {"type": "video", "data_url": "data:video/mp4;base64,AAAA"},
        ]
        with self.assertRaises(MediaInputError):
            prepare_media(items, video_fps=1.0, max_video_frames=8)


if __name__ == "__main__":
    unittest.main()
