-- ============================================================================
-- src/jarvis/core/db/migrations/0021_health.sql
--
-- Migration 0021: HYGIEIA, the health agent.
--
-- Meals are still logged in the life log (life_meals); this adds what was
-- in them. Each eaten meal gets an estimate of its nutrients, kept beside
-- it and deleted with it.
--
-- health_foods is what HYGIEIA has learned about what the owner actually
-- eats and can get: every item it has estimated, where it came from (the
-- mess, a campus shop, outside, his room), per-serving values, and how
-- often he has it. Reusing these keeps "one veg cheese Maggi" the same
-- numbers every time, and lets suggestions come from food he can reach.
--
-- health_menu is the mess menu by weekday and meal; rows with no weekday
-- are served every day. Seeded with the current LNMIIT menu.
-- ============================================================================

CREATE TABLE health_profile (
    key   TEXT PRIMARY KEY,                   -- age, height_cm, weight_kg, goal, diet,
    value TEXT NOT NULL                       -- and any target the owner overrides
) WITHOUT ROWID;

INSERT INTO health_profile (key, value) VALUES
    ('age', '18'), ('sex', 'male'), ('height_cm', '178'),
    ('goal', 'gain_muscle'), ('diet', 'non_veg'), ('training_days', '6');

CREATE TABLE health_meal_nutrition (
    meal_id     TEXT PRIMARY KEY REFERENCES life_meals(id) ON DELETE CASCADE,
    items       TEXT NOT NULL,                -- JSON list of estimated items
    kcal        REAL NOT NULL,
    protein_g   REAL NOT NULL,
    carbs_g     REAL NOT NULL,
    fat_g       REAL NOT NULL,
    fibre_g     REAL NOT NULL,
    sodium_mg   REAL NOT NULL,
    sugar_g     REAL NOT NULL,                -- added sugar
    junk        INTEGER NOT NULL DEFAULT 0,   -- fried / ultra-processed items in it
    confidence  TEXT NOT NULL DEFAULT 'medium'
                CHECK (confidence IN ('low', 'medium', 'high')),
    source      TEXT NOT NULL DEFAULT 'estimate' CHECK (source IN ('estimate', 'owner')),
    created_ts  TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE health_foods (
    key         TEXT PRIMARY KEY,             -- normalised name
    name        TEXT NOT NULL,
    place       TEXT,                         -- mess | campus | outside | room | home
    serving     TEXT NOT NULL,                -- what one serving means: "1 pack (70 g)"
    kcal        REAL NOT NULL,
    protein_g   REAL NOT NULL,
    carbs_g     REAL NOT NULL,
    fat_g       REAL NOT NULL,
    fibre_g     REAL NOT NULL,
    sodium_mg   REAL NOT NULL,
    sugar_g     REAL NOT NULL,
    junk        INTEGER NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT 'learned' CHECK (source IN ('learned', 'menu', 'owner')),
    times_eaten INTEGER NOT NULL DEFAULT 0,
    last_eaten  TEXT,
    updated_ts  TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE health_menu (
    id      INTEGER PRIMARY KEY,
    weekday INTEGER CHECK (weekday BETWEEN 0 AND 6),  -- 0 Monday; NULL = every day
    slot    TEXT NOT NULL CHECK (slot IN ('breakfast', 'lunch', 'snack', 'dinner')),
    items   TEXT NOT NULL                     -- comma-separated, as printed
);

CREATE INDEX idx_health_menu ON health_menu (slot, weekday);

INSERT INTO health_menu (weekday, slot, items) VALUES
    (NULL, 'breakfast', 'Bread (brown/white), Butter or Peanut Butter (limited), Jam (limited), Tea/Coffee/Milk, Bournvita, Fruit (one piece), Cornflakes/Oats'),
    (0, 'breakfast', 'Sprouts, Omelette (limited) or Idli Sambhar, Coconut Chutney'),
    (1, 'breakfast', 'Boiled Chana Chaat, Aloo Paratha, Curd, Garlic Chutney, Ketchup'),
    (2, 'breakfast', 'Sprouts, Boiled Eggs (limited) or Sambhar Vada, Coconut Chutney'),
    (3, 'breakfast', 'Sprouts, Boiled Eggs (limited) or Namkeen Poha, Jalebi'),
    (4, 'breakfast', 'Boiled Moong Chaat, Omelette (limited) or Bedami Puri, Aloo Sabji'),
    (5, 'breakfast', 'Sprouts, Omelette (limited) or Dosa Sambhar, Coconut Chutney'),
    (6, 'breakfast', 'Sprouts, Omelette (limited) or Samosa (limited), Jalebi, Green Chutney, Imli Chutney'),
    (NULL, 'lunch', 'Salad (two of cucumber, onion, tomato, beetroot, radish, carrot), Plain/Butter Roti, Missi Roti'),
    (0, 'lunch', 'Matar Paneer, Dal Arhar, Boondi Raita (limited), Veg Pulao, Lemon Pickle, Fryums (limited)'),
    (1, 'lunch', 'Mix Veg, Dal Makhani, Chhach (limited), Pudina Rice, Mixed Pickle, Coconut Burfi (limited), Papad (limited)'),
    (2, 'lunch', 'Dry Chhole Masala, Dal Fry, Boondi Raita (limited), Schezwan Rice, Mixed Pickle, Angoori Petha (limited), Papad (limited)'),
    (3, 'lunch', 'Aloo Bhujia, Kadhi Pakora, Rooh Afza, Jeera Rice, Lemon Pickle, Papad (limited)'),
    (4, 'lunch', 'Lauki Tamatar, Rajma, Cucumber Raita (limited), Plain Rice, Mixed Pickle, Besan Burfi (limited), Fryums (limited)'),
    (5, 'lunch', 'Bati, Mix Dal, Mirch ka Salan, Boondi Raita (limited), Veg Biryani, Lemon Pickle, Fryums (limited)'),
    (6, 'lunch', 'Punjabi Chhole, Masala Mirch, Lassi (limited), Lemon Rice, Bhature, Mixed Pickle, Papad (limited)'),
    (NULL, 'snack', 'Tea'),
    (0, 'snack', 'Veg Maggi, Cold Coffee'),
    (1, 'snack', 'Vada Pav, Green Chutney, Ketchup, Nimbu Pani'),
    (2, 'snack', 'White Sauce Pasta or Chowmein (alternate weeks), Ketchup, Cold Coffee'),
    (3, 'snack', 'Papdi Chaat, Curd, Green Chutney, Imli Chutney, Rasna'),
    (4, 'snack', 'Stuffed Kulcha, Ketchup, Milk Rose'),
    (5, 'snack', 'Pani Puri with aloo (6 pcs), Jaljeera'),
    (6, 'snack', 'Corn Chaat, Thandai'),
    (NULL, 'dinner', 'Salad (one of cucumber, tomato, beetroot, radish, carrot), Onion, Plain/Butter Roti, Roasted Papad (limited)'),
    (0, 'dinner', 'Fried Aloo Pyaz Sabji, Dal Moong Tadka, Plain Rice, Mixed Pickle, Ice Cream (limited)'),
    (1, 'dinner', 'Black Chana Curry, Dal Urad Tadka, Aloo Pyaz Pulao, Puri, Lemon Pickle, Besan Ladoo (limited)'),
    (2, 'dinner', 'Shahi Paneer, Dal Arhar, Fried Rice, Lemon Pickle, Gulab Jamun (limited)'),
    (3, 'dinner', 'Nutri Keema, Panchratan Dal, Dum Biryani, Lemon Pickle, Rasgulla (limited)'),
    (4, 'dinner', 'Veg Kofta, Akkha Masoor Dal, Tawa Pulao, Mixed Pickle, Shahi Toast (limited)'),
    (5, 'dinner', 'Paneer Do Pyaza, Dal Urad Fry, Jeera Rice, Lemon Pickle, Fruit Custard (limited)'),
    (6, 'dinner', 'Mix Veg, Dal Makhani, Pudina Rice, Mixed Pickle, Sama Rice Kheer (limited)');
