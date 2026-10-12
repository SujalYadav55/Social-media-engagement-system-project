"""
Basic tests. Run from the project folder with:  python -m unittest discover tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app, parse_form, rate_category  # noqa: E402

GOOD_FORM = {
    "followers": "5000",
    "post_type": "reel",
    "caption": "Sunset at the campus lake. Which spot is your favourite? #campus #sunset",
    "day_of_week": "Wednesday",
    "posting_hour": "18",
    "has_cta": "on",
}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_pages_load(self):
        for path in ("/", "/about"):
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_valid_prediction(self):
        response = self.client.post("/predict", data=GOOD_FORM)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predicted engagement rate", response.data)
        self.assertIn(b"Predicted likes", response.data)

    def test_bad_followers_is_rejected(self):
        bad = dict(GOOD_FORM, followers="abc")
        response = self.client.post("/predict", data=bad)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Followers must be a whole number", response.data)

    def test_too_many_hashtags_is_rejected(self):
        bad = dict(GOOD_FORM, caption=" ".join(f"#tag{i}" for i in range(31)))
        response = self.client.post("/predict", data=bad)
        self.assertEqual(response.status_code, 400)

    def test_caption_is_split_into_text_and_hashtags(self):
        _, features, errors, _ = parse_form(GOOD_FORM)
        self.assertEqual(errors, [])
        self.assertEqual(features["hashtags"], 2)
        self.assertEqual(features["caption_length"],
                         len("Sunset at the campus lake. Which spot is your favourite?"))

    def test_rate_categories(self):
        self.assertEqual(rate_category(0.5), "Low")
        self.assertEqual(rate_category(2.0), "Average")
        self.assertEqual(rate_category(4.0), "High")
        self.assertEqual(rate_category(8.0), "Very high")

    def test_more_followers_means_more_likes(self):
        small = self.client.post("/predict", data=dict(GOOD_FORM, followers="1000"))
        large = self.client.post("/predict", data=dict(GOOD_FORM, followers="100000"))
        self.assertNotEqual(small.data, large.data)


if __name__ == "__main__":
    unittest.main()
