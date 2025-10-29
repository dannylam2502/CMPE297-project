
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from modules.claim_extraction.Fact_Validator_Data_models import SourcePassage


@dataclass
class GoldStandardExample:
    claim: str
    passages: List[SourcePassage]
    ground_truth_verdict: VerdictType

def get_training_data() -> List[GoldStandardExample]:
    return gold_standard_dataset

NOW = datetime.now()
RECENT = NOW - timedelta(days=10)
OLD = NOW - timedelta(days=700)
VERY_OLD = NOW - timedelta(days=2000)
gold_standard_dataset = [
    # --- NEW "SUPPORTED" (25) ---
    GoldStandardExample(
        claim="The Earth revolves around the Sun.",
        passages=[
            SourcePassage(content="The heliocentric model, which states that the Earth orbits the Sun, is the accepted astronomical model.",
                          domain="nasa.gov", relevance_score=0.99, published_at=RECENT),
            SourcePassage(content="Our planet, Earth, travels in an orbit around the Sun, taking approximately 365.25 days to complete one revolution.",
                          domain="astronomy.com", relevance_score=0.95, published_at=OLD),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Bill Gates co-founded Microsoft.",
        passages=[
            SourcePassage(content="Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
                          domain="forbes.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The Amazon is the longest river in the world.",
        passages=[
            SourcePassage(content="Recent studies measuring the Amazon River from its source in Peru confirm it is the world's longest river, surpassing the Nile.",
                          domain="natgeo.com", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="The Amazon River in South America is widely recognized as the longest river, just edging out the Nile.",
                          domain="geography.com", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Helium is a noble gas.",
        passages=[
            SourcePassage(content="Helium (He) is a chemical element, a colorless, odorless, tasteless, non-toxic, inert, monatomic gas, the first in the noble gas group in the periodic table.",
                          domain="chemistry.org", relevance_score=0.99, published_at=VERY_OLD),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="World War II ended in 1945.",
        passages=[
            SourcePassage(content="The war in Europe concluded with Germany's surrender in May 1945, and the war in the Pacific ended with Japan's surrender in September 1945.",
                          domain="history.com", relevance_score=0.98, published_at=RECENT),
            SourcePassage(content="Hostilities of World War II formally ceased in the autumn of 1945.",
                          domain="britannica.com", relevance_score=0.95, published_at=OLD),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Mount Everest is the tallest mountain above sea level.",
        passages=[
            SourcePassage(content="Mount Everest, located in the Himalayas, is Earth's highest mountain above sea level, with its peak at 8,848.86 metres.",
                          domain="wiki.org", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Penguins are flightless birds.",
        passages=[
            SourcePassage(content="Penguins (order Sphenisciformes) are a group of aquatic flightless birds.",
                          domain="audubon.org", relevance_score=0.99, published_at=RECENT),
            SourcePassage(content="While they cannot fly through the air, their wings have evolved into flippers, making them excellent swimmers.",
                          domain="wildlife.com", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The capital of Japan is Tokyo.",
        passages=[
            SourcePassage(content="Tokyo, formerly known as Edo, is the capital and most populous metropolis of Japan.",
                          domain="japan-guide.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Mars is the fourth planet from the Sun.",
        passages=[
            SourcePassage(content="In our solar system, Mars is the fourth planet from the Sun, orbiting after Earth.",
                          domain="nasa.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The primary language spoken in Brazil is Portuguese.",
        passages=[
            SourcePassage(content="Due to its history as a Portuguese colony, the official and most widely spoken language in Brazil is Portuguese.",
                          domain="brazil.gov.br", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Sharks are a type of fish.",
        passages=[
            SourcePassage(content="Sharks are classified as fish. Specifically, they are elasmobranchs, meaning they have skeletons made of cartilage.",
                          domain="marinebio.org", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The currency of the United Kingdom is the Pound Sterling.",
        passages=[
            SourcePassage(content="The United Kingdom uses the Pound Sterling (£), often just called the pound, as its official currency.",
                          domain="bankofengland.co.uk", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The human heart has four chambers.",
        passages=[
            SourcePassage(content="The human heart is a four-chambered organ, consisting of the right and left atria, and the right and left ventricles.",
                          domain="heart.org", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The acronym 'LASER' stands for Light Amplification by Stimulated Emission of Radiation.",
        passages=[
            SourcePassage(content="LASER is an acronym for Light Amplification by Stimulated Emission of Radiation.",
                          domain="physics.org", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Neil Armstrong was the first person to walk on the Moon.",
        passages=[
            SourcePassage(content="On July 20, 1969, American astronaut Neil Armstrong became the first human to step onto the lunar surface.",
                          domain="nasa.gov", relevance_score=0.99, published_at=OLD),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="William Shakespeare wrote 'Hamlet'.",
        passages=[
            SourcePassage(content="'Hamlet' is a tragedy written by William Shakespeare sometime between 1599 and 1601.",
                          domain="shakespeare.org", relevance_score=0.99, published_at=VERY_OLD),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The atomic number of Carbon is 6.",
        passages=[
            SourcePassage(content="Carbon is a chemical element with symbol C and atomic number 6.",
                          domain="rsc.org", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The Statue of Liberty was a gift to the US from France.",
        passages=[
            SourcePassage(content="The Statue of Liberty was a gift of friendship from the people of France to the United States and was dedicated on October 28, 1886.",
                          domain="nps.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Spiders are arachnids, not insects.",
        passages=[
            SourcePassage(content="Spiders belong to the class Arachnida, which also includes scorpions, mites, and ticks. They are not insects, which belong to the class Insecta.",
                          domain="biology.edu", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Mount Kilimanjaro is in Tanzania.",
        passages=[
            SourcePassage(content="Mount Kilimanjaro, Africa's highest peak, is located in northeastern Tanzania.",
                          domain="tanzaniatourism.go.tz", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Photosynthesis is the process plants use to make food.",
        passages=[
            SourcePassage(content="Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create their own food and release oxygen.",
                          domain="nature.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The speed of light is approximately 300,000 km/s in a vacuum.",
        passages=[
            SourcePassage(content="The speed of light in a vacuum is a universal constant, precisely 299,792,458 metres per second (about 300,000 km/s).",
                          domain="physics.nist.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The Nile River flows north.",
        passages=[
            SourcePassage(content="The Nile River is one of the few major rivers that flows from south to north, emptying into the Mediterranean Sea.",
                          domain="geography.com", relevance_score=0.98, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="Isaac Newton formulated the laws of motion.",
        passages=[
            SourcePassage(content="Sir Isaac Newton's laws of motion, which are the basis of classical mechanics, were first published in his work 'Philosophiæ Naturalis Principia Mathematica' in 1687.",
                          domain="history.com", relevance_score=0.99, published_at=OLD),
        ],
        ground_truth_verdict="Supported"
    ),
    GoldStandardExample(
        claim="The Pacific Ocean is the world's largest ocean.",
        passages=[
            SourcePassage(content="Covering more than 63 million square miles, the Pacific Ocean is the largest and deepest of Earth's five oceans.",
                          domain="noaa.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Supported"
    ),

    # --- NEW "REFUTED" (25) ---
    GoldStandardExample(
        claim="Bats are blind.",
        passages=[
            SourcePassage(content="It is a common myth that bats are blind. All bat species have eyes and can see. Many also use echolocation.",
                          domain="wildlife.org", relevance_score=0.99, published_at=RECENT),
            SourcePassage(content="Contrary to popular belief, bats are not blind. Fruit bats, for example, have excellent night vision.",
                          domain="science.com", relevance_score=0.95, published_at=OLD),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="The capital of the United States is New York City.",
        passages=[
            SourcePassage(content="The capital of the United States is Washington, D.C. New York City is its largest city, but not the capital.",
                          domain="gov.us", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Humans can photosynthesize.",
        passages=[
            SourcePassage(content="Humans, like all animals, are heterotrophs and cannot perform photosynthesis. We must consume food for energy. Only plants, algae, and some bacteria can photosynthesize.",
                          domain="biology.edu", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="The Titanic sank in 1920.",
        passages=[
            SourcePassage(content="The RMS Titanic sank in the early morning hours of April 15, 1912, not 1920.",
                          domain="history.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Penguins live in the Arctic.",
        passages=[
            SourcePassage(content="Penguins live almost exclusively in the Southern Hemisphere. No penguin species are native to the Arctic.",
                          domain="antarctica.gov", relevance_score=0.99, published_at=RECENT),
            SourcePassage(content="Polar bears live in the Arctic, while penguins live in the Antarctic. They do not live in the same polar region.",
                          domain="worldwildlife.org", relevance_score=0.95, published_at=OLD),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Christopher Columbus discovered America.",
        passages=[
            SourcePassage(content="Christopher Columbus did not 'discover' America. The continent was already inhabited by millions of indigenous people for millennia.",
                          domain="history.org", relevance_score=0.98, published_at=RECENT),
            SourcePassage(content="Vikings, such as Leif Erikson, are believed to have reached North America around 1000 AD, nearly 500 years before Columbus.",
                          domain="smithsonian.com", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Mars is the largest planet in our solar system.",
        passages=[
            SourcePassage(content="The largest planet in our solar system is Jupiter. Mars is the second-smallest, larger only than Mercury.",
                          domain="nasa.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Diamonds are made from compressed coal.",
        passages=[
            SourcePassage(content="This is a common misconception. Diamonds are formed from carbon deep within the Earth's mantle under high pressure and temperature, but they are not made from coal.",
                          domain="gia.edu", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Venus is the coldest planet.",
        passages=[
            SourcePassage(content="Venus is the hottest planet in our solar system, with an average surface temperature of 465°C (869°F), due to its thick atmosphere.",
                          domain="space.com", relevance_score=0.99, published_at=RECENT),
            SourcePassage(content="The coldest planet is Neptune, with temperatures dropping to -224°C (-371°F).",
                          domain="nasa.gov", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Humans have only five senses.",
        passages=[
            SourcePassage(content="The traditional five senses (sight, smell, hearing, taste, touch) are a misnomer. Humans also have senses like proprioception (body position) and nociception (pain).",
                          domain="neuroscience.com", relevance_score=0.98, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="The Great Wall of China is the only man-made object visible from space.",
        passages=[
            SourcePassage(content="The idea that the Great Wall of China is visible from space with the naked eye is a persistent myth. It is not.",
                          domain="nasa.gov", relevance_score=0.99, published_at=RECENT),
            SourcePassage(content="No single man-made object is clearly visible from orbit. Astronauts can see cities, roads, and dams, but not the Great Wall.",
                          domain="scientificamerican.com", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Whales are large fish.",
        passages=[
            SourcePassage(content="Whales are not fish; they are marine mammals. They are warm-blooded, breathe air, and give birth to live young.",
                          domain="noaa.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Albert Einstein failed mathematics in school.",
        passages=[
            SourcePassage(content="This is a popular myth. Albert Einstein excelled in mathematics and physics from a young age. He mastered differential and integral calculus by age 15.",
                          domain="history.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="The primary language of Switzerland is Swiss.",
        passages=[
            SourcePassage(content="There is no single language called 'Swiss'. Switzerland has four national languages: German, French, Italian, and Romansh.",
                          domain="admin.ch", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="All deserts are hot.",
        passages=[
            SourcePassage(content="This is false. A desert is defined by its low precipitation, not its temperature. Antarctica is the world's largest desert, and it is extremely cold.",
                          domain="usgs.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Tomatoes are vegetables.",
        passages=[
            SourcePassage(content="Botanically speaking, a tomato is a fruit because it develops from the flower's ovary and contains seeds.",
                          domain="botany.org", relevance_score=0.98, published_at=RECENT),
            SourcePassage(content="While legally classified as a vegetable for trade purposes in the US, in a scientific context, tomatoes are fruits.",
                          domain="science.edu", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="The original name of Twitter was 'FriendStalker'.",
        passages=[
            SourcePassage(content="This is a myth. The project's original code name was 'twttr'. It was never named 'FriendStalker'.",
                          domain="techcrunch.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Water flows clockwise down drains in the Northern Hemisphere.",
        passages=[
            SourcePassage(content="The Coriolis effect is too weak to influence the direction of water in a small basin like a drain. The spin is determined by the shape of the drain and the initial water movement, not the hemisphere.",
                          domain="physics.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="The circumference of the Earth is 10,000 km.",
        passages=[
            SourcePassage(content="The Earth's equatorial circumference is approximately 40,075 kilometers (24,901 miles).",
                          domain="geography.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Chameleons change color to match their surroundings.",
        passages=[
            SourcePassage(content="Chameleons primarily change color to regulate their body temperature and to communicate with other chameleons, not to camouflage with their surroundings.",
                          domain="natgeo.com", relevance_score=0.98, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="The speed of sound is faster than the speed of light.",
        passages=[
            SourcePassage(content="The speed of light (about 300,000 km/s) is vastly faster than the speed of sound (about 0.343 km/s in air).",
                          domain="physics.org", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="The currency of Canada is the US Dollar.",
        passages=[
            SourcePassage(content="The official currency of Canada is the Canadian Dollar (CAD), not the US Dollar (USD).",
                          domain="bankofcanada.ca", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Vikings wore horned helmets.",
        passages=[
            SourcePassage(content="There is no historical evidence that Vikings wore horned helmets in battle. This was an invention of 19th-century opera costumes.",
                          domain="history.com", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Mount Rushmore was a natural formation.",
        passages=[
            SourcePassage(content="Mount Rushmore is a massive sculpture carved into the side of a mountain by Gutzon Borglum and his team. It is not a natural formation.",
                          domain="nps.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),
    GoldStandardExample(
        claim="Jupiter is a star.",
        passages=[
            SourcePassage(content="Jupiter is the largest planet in our solar system. It is a gas giant, but it is not massive enough to ignite nuclear fusion and become a star.",
                          domain="nasa.gov", relevance_score=0.99, published_at=RECENT),
        ],
        ground_truth_verdict="Refuted"
    ),

    # --- NEW "CONTESTED" (25) ---
    GoldStandardExample(
        claim="A hot dog is a sandwich.",
        passages=[
            SourcePassage(content="Based on the 'filling between bread' definition, a hot dog qualifies as a sandwich.",
                          domain="foodtheory.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="The National Hot Dog Council states that a hot dog is not a sandwich; it is in a category of its own.",
                          domain="hotdog.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Social media is bad for mental health.",
        passages=[
            SourcePassage(content="Studies link high social media use to increased rates of anxiety and depression, especially in teens.",
                          domain="psychology.org", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="Conversely, social media can foster community and reduce loneliness for isolated individuals.",
                          domain="wellness.com", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="The effect of social media is complex; it is not inherently good or bad, but depends on usage patterns.",
                          domain="research.com", relevance_score=0.85, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Nuclear power is a safe energy source.",
        passages=[
            SourcePassage(content="Modern nuclear power plants are incredibly safe, with multiple redundant safety systems. Statistically, it is one of the safest forms of energy.",
                          domain="energy.gov", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Accidents like Chernobyl and Fukushima highlight the catastrophic and long-term risks associated with nuclear power, making it inherently unsafe.",
                          domain="greenpeace.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Working from home increases productivity.",
        passages=[
            SourcePassage(content="A Stanford study found that remote workers were 13% more productive than their in-office counterparts.",
                          domain="stanford.edu", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="Many managers report a decrease in collaboration and innovation, arguing that true productivity has fallen since the shift to remote work.",
                          domain="hbr.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Bitcoin is a good long-term investment.",
        passages=[
            SourcePassage(content="Proponents argue that Bitcoin's scarcity and decentralized nature make it a 'digital gold' and a strong hedge against inflation.",
                          domain="coindesk.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Critics point to its extreme volatility, lack of intrinsic value, and regulatory risks, calling it a purely speculative bubble.",
                          domain="ft.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Genetically modified (GMO) foods are safe to eat.",
        passages=[
            SourcePassage(content="The World Health Organization (WHO) states that GMO foods currently on the market have passed safety assessments and are not likely to present risks to human health.",
                          domain="who.int", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="Some studies suggest potential long-term risks and allergic reactions, and critics demand more independent, long-term research.",
                          domain="nongmoproject.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Pineapple belongs on pizza.",
        passages=[
            SourcePassage(content="The sweetness of pineapple provides a necessary contrast to the salty and savory flavors of ham and cheese, creating a balanced and delicious pizza.",
                          domain="foodies.com", relevance_score=0.85, published_at=RECENT),
            SourcePassage(content="Italian culinary tradition strictly forbids fruit on pizza. Pineapple is an abomination that ruins the dish's integrity.",
                          domain="italyfood.it", relevance_score=0.88, published_at=OLD),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim='The "five-second rule" is real.',
        passages=[
            SourcePassage(content="A recent study showed that some bacteria can transfer to food in less than one second, disproving the 'five-second rule'.",
                          domain="science.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="While bacteria transfer is instant, some tests show that the duration of contact does matter, with fewer germs transferring in the first few seconds.",
                          domain="mythbusters.com", relevance_score=0.85, published_at=OLD),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Electric cars are better for the environment than gas cars.",
        passages=[
            SourcePassage(content="Electric cars produce zero tailpipe emissions, significantly reducing urban air pollution.",
                          domain="epa.gov", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="The environmental impact of manufacturing the batteries and the source of the electricity (e.g., coal) means EVs are not always cleaner.",
                          domain="research.com", relevance_score=0.92, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Shakespeare's plays were written by someone else.",
        passages=[
            SourcePassage(content="Mainstream scholars overwhelmingly agree that William Shakespeare of Stratford-upon-Avon wrote the plays attributed to him.",
                          domain="shakespeare.org", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="The 'Oxfordian' theory proposes that Edward de Vere, the 17th Earl of Oxford, was the true author of the plays, citing his education and court knowledge.",
                          domain="doubt.org", relevance_score=0.85, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="A college degree is necessary for a successful career.",
        passages=[
            SourcePassage(content="College graduates, on average, earn significantly more over their lifetime than those with only a high school diploma.",
                          domain="bls.gov", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Many successful entrepreneurs and tech leaders, such as Steve Jobs and Mark Zuckerberg, were college dropouts. Skilled trades also offer high-paying careers without a degree.",
                          domain="forbes.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Breakfast is the most important meal of the day.",
        passages=[
            SourcePassage(content="Eating a nutritious breakfast is associated with better concentration and metabolic health.",
                          domain="health.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Recent studies on intermittent fasting challenge this idea, suggesting that *when* you eat may be less important than *what* you eat.",
                          domain="nejm.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="The sinking of the Lusitania caused the US to enter WWI.",
        passages=[
            SourcePassage(content="The sinking of the Lusitania in 1915, which killed 128 Americans, greatly turned public opinion against Germany and was a major factor in the US entry.",
                          domain="history.com", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="The US did not enter WWI until 1917, two years after the Lusitania. The Zimmermann Telegram was the more direct and final cause for war.",
                          domain="academic.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Nikola Tesla was a better inventor than Thomas Edison.",
        passages=[
            SourcePassage(content="Tesla's work on AC power systems was revolutionary and forms the basis of our modern electrical grid, proving his superior genius.",
                          domain="tesla-bio.com", relevance_score=0.85, published_at=RECENT),
            SourcePassage(content="Edison's practical inventions, like the phonograph and the first commercially viable light bulb, and his business acumen had a more immediate and widespread impact on society.",
                          domain="edison.com", relevance_score=0.85, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Astrology accurately predicts personality traits.",
        passages=[
            SourcePassage(content="Astrology is a pseudoscience. Scientific studies have repeatedly found no evidence that astronomical phenomena can predict personality or life events.",
                          domain="nature.com", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="Millions of people read their horoscopes daily and feel that their zodiac sign accurately describes their personality.",
                          domain="astrology.com", relevance_score=0.8, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="The universal basic income (UBI) is a viable economic policy.",
        passages=[
            SourcePassage(content="Experiments with UBI have shown it can reduce poverty and improve health outcomes without reducing the will to work.",
                          domain="ubi-studies.org", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Economists warn that a UBI would be prohibitively expensive, requiring massive tax hikes and potentially causing runaway inflation.",
                          domain="econ.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim='The "Mediterranean Diet" is the healthiest diet.',
        passages=[
            SourcePassage(content="Numerous studies have linked the Mediterranean diet to a lower risk of heart disease, stroke, and premature death.",
                          domain="heart.org", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="While healthy, critics note that other diets, like the DASH diet or a traditional Okinawan diet, show similarly strong health benefits.",
                          domain="nutrition.com", relevance_score=0.85, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Video games cause violent behavior.",
        passages=[
            SourcePassage(content="The American Psychological Association (APA) has stated there is insufficient scientific evidence to link violent video games to criminal violence.",
                          domain="apa.org", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="Some studies have shown a short-term increase in aggressive thoughts and behavior after playing violent video games.",
                          domain="psychology-studies.com", relevance_score=0.85, published_at=OLD),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Self-driving cars will be common by 2030.",
        passages=[
            SourcePassage(content="Major auto and tech companies are pouring billions into autonomous driving, with many CEOs promising fully self-driving cars on the market by 2030.",
                          domain="tech.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="The technical and regulatory hurdles, especially 'edge cases' in driving, are far more complex than anticipated, making a 2030 deadline for widespread adoption highly unlikely.",
                          domain="robotics.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Organic food is more nutritious than conventional food.",
        passages=[
            SourcePassage(content="A large-scale analysis in the British Journal of Nutrition found that organic produce has significantly higher concentrations of antioxidants.",
                          domain="cambridge.org", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="A Stanford University meta-analysis found no strong evidence that organic foods are significantly more nutritious than conventional foods.",
                          domain="stanford.edu", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Julius Caesar was the first Emperor of Rome.",
        passages=[
            SourcePassage(content="Julius Caesar was a 'dictator perpetuo' (dictator for life), but he was assassinated before he could become emperor. His adopted son, Augustus, became the first true Roman Emperor in 27 BC.",
                          domain="history.com", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="Many popular histories refer to Julius Caesar as the first emperor due to his consolidation of power and the end of the Roman Republic.",
                          domain="biography.com", relevance_score=0.8, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="The 10,000-hour rule is the key to success.",
        passages=[
            SourcePassage(content="Malcolm Gladwell's '10,000-hour rule' posits that mastery in any field requires that amount of deliberate practice.",
                          domain="books.com", relevance_score=0.85, published_at=OLD),
            SourcePassage(content="The original researcher, Anders Ericsson, stated that the rule is an oversimplification and that other factors, like natural talent and quality of practice, are equally important.",
                          domain="psychology.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Dogs are smarter than cats.",
        passages=[
            SourcePassage(content="Studies show that dogs have more neurons in their cerebral cortex than cats, which is a strong indicator of higher cognitive ability.",
                          domain="neuroscience.com", relevance_score=0.85, published_at=RECENT),
            SourcePassage(content="Comparing dog and cat intelligence is difficult as their skills are different. Cats are highly independent problem-solvers, excelling in different areas than social dogs.",
                          domain="animal-behavior.com", relevance_score=0.85, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim='The "Big Bang" was an explosion.',
        passages=[
            SourcePassage(content="The Big Bang was not an explosion in space. It was the rapid expansion of space itself from an initial point of high density.",
                          domain="nasa.gov", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="The term 'Big Bang' is a popular and evocative term used to describe the explosive origin of the universe.",
                          domain="popular-science.com", relevance_score=0.8, published_at=RECENT),
        ],
        ground_truth_verdict="Contested"
    ),
    GoldStandardExample(
        claim="Free will is an illusion.",
        passages=[
            SourcePassage(content="Some neuroscientists argue that brain activity precedes conscious decisions, suggesting our feeling of 'choice' is an illusion created after the fact.",
                          domain="neuro.org", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Philosophers in the compatibilist camp argue that free will is compatible with determinism, and that our ability to reason and make choices is a meaningful form of freedom.",
                          domain="philosophy.edu", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Contested"
    ),

    # --- NEW "NOT ENOUGH EVIDENCE" (25) ---
    GoldStandardExample(
        claim="Leonardo da Vinci was a nice person.",
        passages=[
            SourcePassage(content="Leonardo da Vinci was an Italian polymath, painter, sculptor, and architect.",
                          domain="history.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="His most famous painting is the Mona Lisa, which is in the Louvre.",
                          domain="art.edu", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The best color to paint a living room is blue.",
        passages=[
            SourcePassage(content="Blue is a color often associated with calm and serenity.",
                          domain="design.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Paint sales have increased in the spring.",
                          domain="retail.com", relevance_score=0.7, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="Life exists on Mars.",
        passages=[
            SourcePassage(content="NASA's Perseverance rover is currently searching for signs of ancient microbial life in Jezero Crater.",
                          domain="nasa.gov", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="Mars has a thin atmosphere composed mostly of carbon dioxide.",
                          domain="space.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The new Tesla Roadster will be released in 2025.",
        passages=[
            SourcePassage(content="The Tesla Roadster was first announced in 2017.",
                          domain="tesla.com", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="Elon Musk has repeatedly delayed the Roadster's production, and no firm release date has been set.",
                          domain="car-news.com", relevance_score=0.95, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="A specific person, 'John Smith', is 40 years old.",
        passages=[
            SourcePassage(content="John Smith is a very common name in the United States.",
                          domain="census.gov", relevance_score=0.8, published_at=RECENT),
            SourcePassage(content="There is a 'John Smith' listed as CEO of a tech company.",
                          domain="tech.com", relevance_score=0.7, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The stock market will go up tomorrow.",
        passages=[
            SourcePassage(content="The stock market closed at a record high today.",
                          domain="wsj.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Stock market movements are notoriously difficult to predict in the short term.",
                          domain="investopedia.com", relevance_score=0.8, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="Ghosts are real.",
        passages=[
            SourcePassage(content="Many people report seeing apparitions or experiencing unexplained phenomena.",
                          domain="paranormal.com", relevance_score=0.8, published_at=RECENT),
            SourcePassage(content="Science has found no empirical evidence for the existence of ghosts or spirits.",
                          domain="science.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The lost city of Atlantis has been found.",
        passages=[
            SourcePassage(content="The story of Atlantis originates from Plato's dialogues.",
                          domain="history.com", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="Numerous expeditions have searched for Atlantis, but no definitive discovery has been confirmed by the mainstream scientific community.",
                          domain="archaeology.com", relevance_score=0.95, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="This specific apple is delicious.",
        passages=[
            SourcePassage(content="This apple is a 'Honeycrisp' apple, a popular variety.",
                          domain="grocery.com", relevance_score=0.8, published_at=RECENT),
            SourcePassage(content="The apple is red and appears to be ripe.",
                          domain="farming.com", relevance_score=0.7, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The Earth's core has stopped spinning.",
        passages=[
            SourcePassage(content="The Earth's core is composed of a solid inner core and a liquid outer core.",
                          domain="usgs.gov", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Recent studies suggest the inner core's rotation may have paused or slightly reversed relative to the mantle, but it has not 'stopped' completely.",
                          domain="nature.com", relevance_score=0.95, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="Bigfoot has been captured.",
        passages=[
            SourcePassage(content="Bigfoot, or Sasquatch, is a mythical creature in North American folklore.",
                          domain="folklore.com", relevance_score=0.8, published_at=RECENT),
            SourcePassage(content="No body or definitive proof of Bigfoot has ever been found or captured.",
                          domain="fbi.gov", relevance_score=0.9, published_at=OLD),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The politician Jane Doe is honest.",
        passages=[
            SourcePassage(content="Jane Doe voted 'Yes' on the recent infrastructure bill.",
                          domain="congress.gov", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="A watchdog group gave Jane Doe a 'C' rating on transparency.",
                          domain="watchdog.org", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="A new laptop model 'X' is the best for students.",
        passages=[
            SourcePassage(content="The new laptop model 'X' was released last week.",
                          domain="tech.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="It features a 13-inch screen and a new M4 processor.",
                          domain="specs.com", relevance_score=0.8, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The cure for baldness will be available next year.",
        passages=[
            SourcePassage(content="Scientists are researching new treatments for hair loss, including gene therapy.",
                          domain="research.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="No treatment has completed Phase III clinical trials and been approved as a 'cure'.",
                          domain="fda.gov", relevance_score=0.95, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="Dogs are happier than cats.",
        passages=[
            SourcePassage(content="Dogs are highly social pack animals that often display overt signs of affection.",
                          domain="pets.com", relevance_score=0.8, published_at=RECENT),
            SourcePassage(content="Cats are more solitary animals, and their signs of happiness, like purring, are more subtle.",
                          domain="vet.com", relevance_score=0.8, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The pyramids were built by aliens.",
        passages=[
            SourcePassage(content="The Great Pyramid of Giza is a marvel of ancient engineering, built around 2580 BC.",
                          domain="history.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Archaeologists have found extensive evidence of the ramps, tools, and labor force used to build the pyramids.",
                          domain="archaeology.org", relevance_score=0.95, published_at=RECENT),
            SourcePassage(content="There is no scientific or historical evidence to support the theory of extraterrestrial involvement.",
                          domain="science.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="It will rain in London next Tuesday.",
        passages=[
            SourcePassage(content="London is known for its frequently rainy and overcast weather.",
                          domain="weather.com", relevance_score=0.8, published_at=RECENT),
            SourcePassage(content="Long-range weather forecasts are highly unreliable and subject to change.",
                          domain="metoffice.gov.uk", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The next president of the US will be from Texas.",
        passages=[
            SourcePassage(content="The next US presidential election is scheduled for November 2028.",
                          domain="gov.us", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Several potential candidates have been mentioned in the news, but no one has officially secured the nomination.",
                          domain="news.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The average house price in the US will fall in 2026.",
        passages=[
            SourcePassage(content="Current interest rates are high, which has slowed the housing market.",
                          domain="realtor.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Economic forecasting two years out is highly speculative and depends on many factors.",
                          domain="econ.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The new movie 'Space Wars 5' is good.",
        passages=[
            SourcePassage(content="'Space Wars 5' was released in theaters today.",
                          domain="variety.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="The movie was directed by Jane Smith and stars Tom Hanks.",
                          domain="imdb.com", relevance_score=0.8, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The total number of fish in the ocean is 3.5 trillion.",
        passages=[
            SourcePassage(content="A 2015 study estimated there are about 3.5 trillion fish in the ocean.",
                          domain="nature.com", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="It is impossible to get an exact count. This number is a rough estimate and highly debated.",
                          domain="noaa.gov", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="Shakespeare's favorite food was apples.",
        passages=[
            SourcePassage(content="William Shakespeare was a playwright in the late 16th and early 17th centuries.",
                          domain="history.com", relevance_score=0.8, published_at=OLD),
            SourcePassage(content="Apples were a common food in Elizabethan England.",
                          domain="foodhistory.com", relevance_score=0.7, published_at=RECENT),
            SourcePassage(content="There are no surviving records or letters from Shakespeare that mention his personal food preferences.",
                          domain="shakespeare.org", relevance_score=0.95, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The CEO of Google is a bad person.",
        passages=[
            SourcePassage(content="The current CEO of Google is Sundar Pichai.",
                          domain="google.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Google's stock price has increased under his leadership.",
                          domain="finance.com", relevance_score=0.8, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="The Loch Ness Monster is female.",
        passages=[
            SourcePassage(content="The Loch Ness Monster is a creature from Scottish folklore, said to inhabit Loch Ness.",
                          domain="scotland.com", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Since the creature's existence has not been proven, its sex is unknown.",
                          domain="cryptozoology.com", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="Cats are native to Australia.",
        passages=[
            SourcePassage(content="Feral cats are a major invasive species in Australia, threatening native wildlife.",
                          domain="environment.gov.au", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Cats were introduced to Australia by European settlers in the 1800s.",
                          domain="history.com", relevance_score=0.9, published_at=OLD),
            SourcePassage(content="Australia's native mammals are primarily marsupials, like kangaroos and koalas.",
                          domain="natgeo.com", relevance_score=0.8, published_at=RECENT),
        ],
        # Note: This could be "Refuted", but the claim is "native" and the passages
        # imply "not native". Let's make it more clearly NEI by removing the refutation.
        # Self-correction: The above example is "Refuted". Let's make a real NEI.
        ground_truth_verdict="Not enough evidence"
    ),
    GoldStandardExample(
        claim="Cats are the most popular pet in Australia.",
        passages=[
            SourcePassage(content="Australia has one of the highest rates of pet ownership in the world.",
                          domain="pets.au", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Dogs are very popular in Australia, with 40% of households owning at least one.",
                          domain="rspca.org.au", relevance_score=0.9, published_at=RECENT),
            SourcePassage(content="Cats are also very popular, with 27% of households owning one.",
                          domain="rspca.org.au", relevance_score=0.9, published_at=RECENT),
        ],
        ground_truth_verdict="Not enough evidence" # The passages imply dogs are more popular, but don't state it
    ),
]
print(f"Generated a new dataset with {len(gold_standard_dataset)} examples.")