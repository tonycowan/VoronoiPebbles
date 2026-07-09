from procedural_pebbles import PebbleGenerator

gen = PebbleGenerator(
    width=100,
    height=100,
    seed=42,
)

gen.generate(120)

gen.relax(3)

regions, vertices = gen.voronoi()

print(
    f"{len(regions)} regions"
)

print(
    f"{len(vertices)} vertices"
)
