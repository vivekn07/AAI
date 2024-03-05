import enum
import random

# Define Kid enum
class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

# Define function to return a random Kid
def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

# Initialize counters
both_girls = 0
older_girl = 0
either_girl = 0

# Set seed for reproducibility
random.seed(0)

# Simulate 10,000 families
for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

# Print results
print("older girl:", older_girl)
print("both girls:", both_girls)
print("either girl:", either_girl)
print("P(both | older):", both_girls / older_girl)
print("P(both | either):", both_girls / either_girl)
