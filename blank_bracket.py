bracket_pairs = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]
blank = "________________"
for region in ["East", "West", "South", "Midwest"]:
    print(f"\n===== {region} =====")
    print("Round of 64:")
    for a, b in bracket_pairs:
        print(f"  ({a:>2}) {blank}  vs  ({b:>2}) {blank}")
    print("Round of 32:")
    for i in range(4):
        print(f"  {blank}  vs  {blank}")
    print("Sweet 16:")
    for i in range(2):
        print(f"  {blank}  vs  {blank}")
    print("Elite 8:")
    print(f"  {blank}  vs  {blank}")
    print(f"Region champion: {blank}")

print("\n===== Final Four =====")
print(f"  East {blank}  vs Midwest {blank}")
print(f"  South {blank}  vs  West {blank}")
print(f"National champion: {blank}")