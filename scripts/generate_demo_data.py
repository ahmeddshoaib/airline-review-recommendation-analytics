"""Generate synthetic airline-review data for public pipeline validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)
N = 6000

airlines = np.array([f"Airline {index:02d}" for index in range(1, 51)])
airline = RNG.choice(airlines, size=N)
latent = RNG.normal(0, 1, size=N)
value = np.clip(np.rint(3 + 1.1 * latent + RNG.normal(0, 0.7, N)), 1, 5)
ground = np.clip(np.rint(3 + 0.9 * latent + RNG.normal(0, 0.8, N)), 1, 5)
cabin = np.clip(np.rint(3.2 + 0.8 * latent + RNG.normal(0, 0.8, N)), 1, 5)
seat = np.clip(np.rint(3 + 0.6 * latent + RNG.normal(0, 0.9, N)), 1, 5)
food = np.clip(np.rint(2.8 + 0.5 * latent + RNG.normal(0, 1.0, N)), 1, 5)
entertainment = np.clip(np.rint(2.7 + 0.4 * latent + RNG.normal(0, 1.0, N)), 1, 5)
wifi = np.clip(np.rint(2.2 + 0.3 * latent + RNG.normal(0, 1.1, N)), 1, 5)
overall = np.clip(np.rint(5.5 + 2.1 * latent + RNG.normal(0, 1.2, N)), 1, 10)

logit = -1.3 + 0.8 * (value - 3) + 0.45 * (ground - 3) + 0.35 * (cabin - 3) + 0.15 * (overall - 5)
probability = 1 / (1 + np.exp(-logit))
recommended = RNG.binomial(1, probability)

positive = np.array([
    "good value and helpful cabin crew",
    "smooth journey friendly service worth the fare",
    "efficient ground team and comfortable flight",
])
negative = np.array([
    "poor value delay and unhelpful ground service",
    "disappointing experience uncomfortable and expensive",
    "long delay weak communication would not recommend",
])
aligned_text = RNG.random(N) < 0.78
positive_text = RNG.choice(positive, N)
negative_text = RNG.choice(negative, N)
review = np.where(
    np.where(aligned_text, recommended == 1, recommended == 0),
    positive_text,
    negative_text,
)

frame = pd.DataFrame(
    {
        "Airline_Name": airline,
        "Review": review,
        "Recommended": np.where(recommended == 1, "yes", "no"),
        "Overall_Rating": overall,
        "Seat_Comfort": seat,
        "Cabin_Staff_Service": cabin,
        "Food_and_Beverages": food,
        "Ground_Service": ground,
        "Inflight_Entertainment": entertainment,
        "Wifi_and_Connectivity": wifi,
        "Value_For_Money": value,
        "Verified": RNG.integers(0, 2, N),
        "Seat_Type": RNG.choice(["Economy", "Premium Economy", "Business"], N, p=[0.72, 0.12, 0.16]),
    }
)

for column, rate in {
    "Wifi_and_Connectivity": 0.55,
    "Inflight_Entertainment": 0.35,
    "Food_and_Beverages": 0.22,
    "Ground_Service": 0.08,
}.items():
    frame.loc[RNG.random(N) < rate, column] = np.nan

path = Path("data/demo_airline_reviews.csv")
path.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(path, index=False)
print(f"Wrote {len(frame):,} synthetic reviews to {path}")
