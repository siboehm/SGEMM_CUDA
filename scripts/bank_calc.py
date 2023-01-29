banks_naive = lambda r, c: (r * 32 + c) % 32
banks_one_extra = lambda r, c: (r * 33 + c) % 32

ITEMS_PER_WARP = 8


def printBankConflicts(bank_fun):
    for c in range(1):
        banks = []
        for i in range(32):
            row = (i * ITEMS_PER_WARP) // 16 
            col = (i * ITEMS_PER_WARP + c) % 16
            banks.append((i, row, col, bank_fun(row, col)))
        print("Step", c, "\n", "\n".join(["(" + ",".join(str(x) for x in i) + ")" for i in banks]))
        d = {k: 0 for k in range(32)}
        for i in banks:
            d[i[-1]] += 1

        count = 0
        for key, val in d.items():
            if val > 0:
                count += 1

        print(
            f"Bank conflicts (Step {c}): {sorted(d.items(), key=lambda item: item[1], reverse=True)[0][1]}, banks accessed: {count}/32\n"
        )


print("---NAIVE---")
printBankConflicts(banks_naive, 32)

print("\n---EXTRA COL---")
printBankConflicts(banks_one_extra, 33)
