"""A built-in general-knowledge probe pack for C2 (calibration_drift).

Each item is a ``(prompt, gold)`` pair where ``gold`` is the next few
tokens a competent base model should assign high probability to. The
items are deliberately *factually trivial* — the point isn't "does the
model know this?" but "did the fine-tune forget this?" — so the pack
skews toward grade-school geography, chemistry, arithmetic, and
high-frequency idiom.

**Provenance.** All items are public-domain grade-school facts, common
English idioms, or trivially-derivable arithmetic. Nothing here is
sourced from a specific licensed dataset (no TriviaQA / SQuAD / OpenBookQA
text); items were composed by hand from primary-school curricula in
common use across English-speaking countries. This keeps the wheel
license-clean and lets us ship the pack without attribution.

**Per-section origins** (F18 audit trail). Granular provenance is
tracked at the section boundary below rather than per-item — individual
facts like "The capital of France is Paris" are not copyrightable, so
a row-level citation is paperwork without legal substance. If a future
DMCA-style question surfaces on a specific section, the origin
category below narrows the audit to the right primary-school domain.

- Geography — country/capital, ocean, mountain, river, continent facts
  from primary-school geography curricula.
- Natural sciences — physics/chemistry/biology facts at the 4th–6th
  grade level; units + constants from introductory physics.
- Arithmetic — items mechanically derivable (addition, multiplication,
  squares/cubes, conversions). Not a memorization test.
- Language and idiom — high-frequency English idioms in continuous use
  since pre-1928; phrases listed in Merriam-Webster / OED as public-
  domain idiom.
- History — historical dates from standard primary-school history.
  Names (figures, countries) are facts, not creative expression.
- Biology — anatomy and natural-history facts at the 4th–6th grade
  level.
- Technology — basic computing/internet vocabulary from primary-school
  digital-literacy (HTML = "Hypertext Markup Language", etc.).
- Miscellaneous trivia — mixed-domain primary-school facts that didn't
  fit the category rubric above.

**Size.** 200 items. With ``regression_nats=1.0`` and
``assert_fraction_regressed_lt=0.15``, a single regressed item moves
the fraction by 0.5 percentage points — well below the gate's
resolution. This was the B12 fix: the original 30-item pack moved
3.3 pp per regression, making the 15% gate noisy.

**Subsetting.** Pass ``pack_sample: int`` in the spec to take the
first ``N`` items (the order is curated for diversity, not random).
"""

from __future__ import annotations

from typing import Final

CalibrationItem = tuple[str, str]

BUILT_IN_PACK: Final[tuple[CalibrationItem, ...]] = (
    # --- Geography (30) ---
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Russia is", " Moscow"),
    ("The capital of Egypt is", " Cairo"),
    ("The capital of Australia is", " Canberra"),
    ("The capital of Canada is", " Ottawa"),
    ("The capital of Brazil is", " Brasilia"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The smallest continent is", " Australia"),
    ("The largest continent is", " Asia"),
    ("Mount Everest is located on the border of Nepal and", " China"),
    ("The longest river in South America is the", " Amazon"),
    ("The longest river in Africa is the", " Nile"),
    ("The Sahara Desert is in", " Africa"),
    ("The Great Barrier Reef is off the coast of", " Australia"),
    ("Mount Fuji is in", " Japan"),
    ("The Eiffel Tower is in", " Paris"),
    ("Big Ben is in", " London"),
    ("The Statue of Liberty is in", " New York"),
    ("The Colosseum is in", " Rome"),
    ("The pyramids of Giza are in", " Egypt"),
    ("The Andes mountains are in South", " America"),
    ("The Mediterranean Sea borders southern", " Europe"),
    ("The Atlantic Ocean separates the Americas from", " Europe"),
    ("Iceland is an island in the North", " Atlantic"),
    ("Madagascar is an island off the coast of", " Africa"),
    ("Antarctica is the coldest", " continent"),
    # --- Natural sciences (30) ---
    ("Water freezes at zero degrees", " Celsius"),
    ("Water boils at one hundred degrees", " Celsius"),
    ("The chemical symbol for gold is", " Au"),
    ("The chemical symbol for silver is", " Ag"),
    ("The chemical symbol for iron is", " Fe"),
    ("The chemical symbol for sodium is", " Na"),
    ("The chemical symbol for oxygen is", " O"),
    ("The chemical symbol for hydrogen is", " H"),
    ("The chemical symbol for carbon is", " C"),
    ("Light travels faster than", " sound"),
    ("Plants convert sunlight into energy through", " photosynthesis"),
    ("The Earth orbits around the", " Sun"),
    ("The Moon orbits around the", " Earth"),
    ("There are eight planets in our solar", " system"),
    ("The closest star to Earth is the", " Sun"),
    ("The fastest land animal is the", " cheetah"),
    ("The largest mammal on Earth is the blue", " whale"),
    ("Spiders have eight", " legs"),
    ("Insects have six", " legs"),
    ("A baby cat is called a", " kitten"),
    ("A baby dog is called a", " puppy"),
    ("Bees produce", " honey"),
    ("Cows produce", " milk"),
    ("Sound is measured in units called", " decibels"),
    ("Temperature can be measured with a", " thermometer"),
    ("The boiling point of water at sea level is one hundred degrees", " Celsius"),
    ("Atoms are made of protons, neutrons, and", " electrons"),
    ("DNA stands for deoxyribonucleic", " acid"),
    ("The force that pulls objects toward Earth is called", " gravity"),
    ("A rainbow has the colors red, orange, yellow, green, blue, indigo, and", " violet"),
    # --- Arithmetic (20) ---
    ("Two plus two equals", " four"),
    ("Three plus three equals", " six"),
    ("Five plus five equals", " ten"),
    ("Ten times ten equals", " one hundred"),
    ("Half of one hundred is", " fifty"),
    ("A dozen means", " twelve"),
    ("A century is one hundred", " years"),
    ("A millennium is one thousand", " years"),
    ("A decade is ten", " years"),
    ("A score is twenty", " years"),
    ("Six times seven equals forty", "-two"),
    ("Nine times nine equals eighty", "-one"),
    ("Twelve times twelve equals one hundred forty", "-four"),
    ("One half plus one half equals", " one"),
    ("Pi is approximately three point one four", " one"),
    ("There are sixty seconds in a", " minute"),
    ("There are sixty minutes in an", " hour"),
    ("There are twenty-four hours in a", " day"),
    ("There are twelve months in a", " year"),
    ("A right angle is ninety", " degrees"),
    # --- Language and idiom (30) ---
    ("A rose by any other name would smell as", " sweet"),
    ("To be or not to be, that is the", " question"),
    ("The early bird catches the", " worm"),
    ("Actions speak louder than", " words"),
    ("A picture is worth a thousand", " words"),
    ("When in Rome, do as the Romans", " do"),
    ("Better late than", " never"),
    ("All that glitters is not", " gold"),
    ("Birds of a feather flock", " together"),
    ("Don't count your chickens before they", " hatch"),
    ("Don't put all your eggs in one", " basket"),
    ("Every cloud has a silver", " lining"),
    ("Honesty is the best", " policy"),
    ("Look before you", " leap"),
    ("Practice makes", " perfect"),
    ("Rome wasn't built in a", " day"),
    ("The pen is mightier than the", " sword"),
    ("Time flies when you're having", " fun"),
    ("Two heads are better than", " one"),
    ("You can't judge a book by its", " cover"),
    ("Where there's a will, there's a", " way"),
    ("A stitch in time saves", " nine"),
    ("Curiosity killed the", " cat"),
    ("Easier said than", " done"),
    ("Fortune favors the", " bold"),
    ("If at first you don't succeed, try, try", " again"),
    ("It takes two to", " tango"),
    ("Knowledge is", " power"),
    ("Necessity is the mother of", " invention"),
    ("The grass is always greener on the other", " side"),
    # --- History (25) ---
    ("World War II ended in the year", " 1945"),
    ("World War I began in the year", " 1914"),
    ("The first president of the United States was", " George Washington"),
    ("The Berlin Wall fell in", " 1989"),
    ("The American Declaration of Independence was signed in", " 1776"),
    ("Christopher Columbus reached the Americas in", " 1492"),
    ("The French Revolution began in", " 1789"),
    ("The Magna Carta was signed in", " 1215"),
    ("The Roman Empire fell in", " 476"),
    ("The Renaissance began in", " Italy"),
    ("The Industrial Revolution began in", " Britain"),
    ("Isaac Newton discovered the laws of", " motion"),
    ("Albert Einstein developed the theory of", " relativity"),
    ("Marie Curie discovered the elements polonium and", " radium"),
    ("Charles Darwin proposed the theory of", " evolution"),
    ("Alexander Graham Bell invented the", " telephone"),
    ("Thomas Edison invented the light", " bulb"),
    ("The Wright brothers built the first", " airplane"),
    ("Neil Armstrong walked on the Moon in", " 1969"),
    ("The Eiffel Tower was completed in", " 1889"),
    ("The Titanic sank in", " 1912"),
    ("The Great Wall of China was built to defend against northern", " invaders"),
    ("Cleopatra was the last pharaoh of ancient", " Egypt"),
    ("Julius Caesar was a Roman", " general"),
    ("Napoleon Bonaparte was emperor of", " France"),
    # --- Biology (20) ---
    ("Humans have twenty", " fingers and toes"),
    ("The human body has two", " lungs"),
    ("Blood is pumped through the body by the", " heart"),
    ("Humans have thirty-two adult", " teeth"),
    ("The human skeleton has approximately two hundred and six", " bones"),
    ("The largest organ in the human body is the", " skin"),
    ("Humans have five", " senses"),
    ("The brain is part of the central nervous", " system"),
    ("Red blood cells carry", " oxygen"),
    ("The pancreas produces", " insulin"),
    ("Bones are connected at", " joints"),
    ("The eye sees light through the", " pupil"),
    ("The ear contains a small bone called the", " stirrup"),
    ("A human heart has four", " chambers"),
    ("The food we eat is broken down in the", " stomach"),
    ("Vitamin D is produced when skin is exposed to", " sunlight"),
    ("Trees produce oxygen and absorb carbon", " dioxide"),
    ("A caterpillar transforms into a", " butterfly"),
    ("Frogs begin life as", " tadpoles"),
    ("Bears hibernate during the", " winter"),
    # --- Technology (20) ---
    ("HTML stands for HyperText", " Markup Language"),
    ("The World Wide Web was invented by Tim", " Berners-Lee"),
    ("HTTP stands for HyperText Transfer", " Protocol"),
    ("URL stands for Uniform Resource", " Locator"),
    ("RAM stands for Random Access", " Memory"),
    ("CPU stands for Central Processing", " Unit"),
    ("GPU stands for Graphics Processing", " Unit"),
    ("USB stands for Universal Serial", " Bus"),
    ("WiFi is a wireless networking", " technology"),
    ("Email is short for electronic", " mail"),
    ("Software is a set of", " instructions"),
    ("A computer program is written in a programming", " language"),
    ("Python is a popular programming", " language"),
    ("JavaScript is widely used in web", " browsers"),
    ("A pixel is a tiny dot on a", " screen"),
    ("A keyboard is used to type", " characters"),
    ("A mouse is used to point and", " click"),
    ("Bluetooth is a short-range wireless", " technology"),
    ("GPS stands for Global Positioning", " System"),
    ("AI stands for artificial", " intelligence"),
    # --- Miscellaneous trivia (25) ---
    ("One year has", " 365 days"),
    ("A leap year has", " 366 days"),
    ("A week has seven", " days"),
    ("There are seven colors in a", " rainbow"),
    ("There are seven continents on", " Earth"),
    ("There are five Great Lakes in North", " America"),
    ("The Olympic Games are held every four", " years"),
    ("A triangle has three", " sides"),
    ("A square has four", " sides"),
    ("A pentagon has five", " sides"),
    ("A hexagon has six", " sides"),
    ("An octagon has eight", " sides"),
    ("There are twelve signs of the", " zodiac"),
    ("A tripod has three", " legs"),
    ("A bicycle has two", " wheels"),
    ("A piano has eighty-eight", " keys"),
    ("A standard deck has fifty-two", " cards"),
    ("Music is written on a", " staff"),
    ("Notes on the musical scale are do, re, mi, fa, sol, la, and", " ti"),
    ("The primary colors are red, yellow, and", " blue"),
    ("The opposite of black is", " white"),
    ("The opposite of hot is", " cold"),
    ("The opposite of fast is", " slow"),
    ("The opposite of up is", " down"),
    ("The opposite of beginning is", " end"),
)
"""200 items spanning geography, natural sciences, arithmetic, language
& idiom, history, biology, technology, and general trivia. Hand-curated
from public-domain grade-school facts; no third-party dataset license
attaches."""

assert len(BUILT_IN_PACK) == 200, (
    f"BUILT_IN_PACK should have exactly 200 items; got {len(BUILT_IN_PACK)}"
)
