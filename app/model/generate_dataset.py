import pandas as pd
import random
from faker import Faker

fake = Faker()

def generate_apa_reference():
    authors = f"{fake.last_name()}, {fake.first_name()[0]}., & {fake.last_name()}, {fake.first_name()[0]}."
    year = fake.year()
    title = fake.sentence(nb_words=6).rstrip('.')
    journal = fake.bs().title()
    volume = random.randint(1, 30)
    issue = random.randint(1, 10)
    pages = f"{random.randint(1, 200)}-{random.randint(201, 300)}"
    doi = f"https://doi.org/10.{random.randint(1000, 9999)}/xyz.{random.randint(100, 999)}"
    return f"{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}. {doi}"

def generate_non_apa_reference():
    # Randomly select a non-APA format (e.g., MLA, Chicago, or incorrect APA)
    style = random.choice(["MLA", "Chicago", "Bad_APA"])
    if style == "MLA":
        return f"{fake.last_name()}, {fake.first_name()}. \"{fake.sentence(nb_words=6)}\" {fake.bs().title()}, {fake.year()}, pp. {random.randint(1, 200)}-{random.randint(201, 300)}."
    elif style == "Chicago":
        return f"{fake.last_name()}, {fake.first_name()}. {fake.year()}. \"{fake.sentence(nb_words=6)}.\" {fake.bs().title()} {random.randint(1, 30)}, no. {random.randint(1, 10)}: {random.randint(1, 300)}."
    else:  # Bad APA (e.g., missing parentheses, wrong order)
        return f"{fake.last_name()}, {fake.first_name()}. {fake.year()}. {fake.sentence(nb_words=6)}. {fake.bs().title()}, {random.randint(1, 30)}({random.randint(1, 10)}), {random.randint(1, 300)}."

# Generate dataset
data = []
for _ in range(2000):
    if random.random() > 0.5:  # 50% APA, 50% non-APA
        data.append([generate_apa_reference(), "APA"])
    else:
        data.append([generate_non_apa_reference(), "Not APA"])

df = pd.DataFrame(data, columns=["reference_text", "label"])
df.to_csv("apa_dataset_2000.csv", index=False)
print("Dataset generated as 'apa_dataset_2000.csv'")