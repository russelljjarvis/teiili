import numpy as np

def create_lfsr(num_bits):
    lfsr_num_bits = np.unique(num_bits)
    lfsr_len = [int(2**n - 1) for n in lfsr_num_bits]

    mask = {3: 0b1011, 5: 0b100101, 6:0b1000011, 9: 0b1000010001}

    lfsr = {}
    lfsr['array'] = []
    init_map = {}
    prev_position = 0
    prev_nbits = 0

    for i, n_bits in enumerate(lfsr_num_bits):
        # Determines possition of lfsr['array'] where each lfsr starts
        init_map[n_bits] = prev_position + (2**prev_nbits - 1)
        prev_position = init_map[n_bits]
        prev_nbits = n_bits

        # Create LFSR values
        lfsr_val = 1
        lfsr['array'].append(lfsr_val)
        for _ in range(1, lfsr_len[i]):
            lfsr_val = lfsr_val << 1
            overflow = lfsr_val >> int(n_bits)
            if overflow:
                lfsr_val ^= mask[n_bits]
            lfsr['array'].append(lfsr_val)

    # Creates values for each element
    lfsr['seed'] = [np.random.randint(2**x-1) for x in num_bits]
    lfsr['max_value'] = [2**x-1 for x in num_bits]
    lfsr['init'] = [init_map[x] for x in num_bits]

    return lfsr
