from procedural_pebbles import PebbleGenerator

gen = PebbleGenerator(
    width=100,
    height=100,
    seed=42,
)

gen.generate(150)

print(gen.point_count)
