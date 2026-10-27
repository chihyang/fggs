import unittest
import torch
from torch_semiring_einsum import einsum, compile_equation, log_viterbi_einsum_forward, AUTOMATIC_BLOCK_SIZE
from fggs.semirings import RealSemiring, ViterbiSemiring
from fggs.equation import *
from typing import List

class TestEquation(unittest.TestCase):

    def assertTEqual(self, input: torch.Tensor, other: torch.Tensor) -> None:
        self.assertTrue(torch.equal(input, other), (input, other))

    def test_einsum_expand_general(self):
        eq = 'jki,bka->ib'

        si = 2
        sj = 3
        sk = 5
        sa = 4
        sb = 2

        ten_a = [
            torch.tensor([0.1, 0.2])
        ]

        ten_b = [
            torch.tensor([10, 20, 40, 30])
        ]

        expected_equations = [
            [
                "i,a->i"
            ]
        ]

        expected_output_shape = [2, 2]

        expected_unsqueeze = [
            [
                [1]
            ]
        ]

        for i, a in enumerate(ten_a):
            for j, b in enumerate(ten_b):
                ea = a.expand(*[sj, sk, si])
                eb = b.expand(*[sb, sk, sa])
                self.helperTestReduceEquation(
                    eq, ea, eb, expected_equations[i][j],
                    expected_output_shape, expected_unsqueeze[i][j])

                self.helperTestReducedEinsum(
                    eq, [a, b], [[sj, sk, si], [sb, sk, sa]])


    def test_einsum_expand_free(self):
        eq = 'ij,ji->ij'

        si = 2
        sj = 3

        ten_a = [
            torch.tensor(0.1),
            torch.tensor([0.1]),
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([[0.1]]),
            torch.tensor([[0.1], [0.2]]),
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        ]

        ten_b = [
            torch.tensor(1),
            torch.tensor([1]),
            torch.tensor([1, 10]),
            torch.tensor([[1]]),
            torch.tensor([[1], [10], [100]]),
            torch.tensor([[1, 10]]),
            torch.tensor([[1, 10], [2, 20], [3, 30]])
        ]

        expected_equations = [
            # []
            [
                ",->",
                ",->",
                ",a->a",
                ",->",
                ",a->a",
                ",a->a",
                ",ab->ba",
            ],
            # [1]
            [
                ",->",
                ",->",
                ",a->a",
                ",->",
                ",a->a",
                ",a->a",
                ",ab->ba",
            ],
            # [sj]
            [
                "a,->a",
                "a,->a",
                "a,b->ba",
                "a,->a",
                "a,a->a",
                "a,b->ba",
                "a,ab->ba",
            ],
            # [1,1]
            [
                ",->",
                ",->",
                ",a->a",
                ",->",
                ",a->a",
                ",a->a",
                ",ab->ba",
            ],
            # [si,1]
            [
                "a,->a",
                "a,->a",
                "a,a->a",
                "a,->a",
                "a,b->ab",
                "a,a->a",
                "a,ba->ab",  # None,
            ],
            # [1,sj]
            [
                "a,->a",
                "a,->a",
                "a,b->ba",
                "a,->a",
                "a,a->a",
                "a,b->ba",
                "a,ab->ba",
            ],
            # [si,sj]
            [
                "ab,->ab",
                "ab,->ab",
                "ab,a->ab",
                "ab,->ab",
                "ab,b->ab",
                "ab,a->ab",
                "ab,ba->ab",  # None,
            ]
        ]

        expected_output_shape = [2, 3]

        expected_unsqueeze = [
            # []
            [
                [0, 1],
                [0, 1],
                [1],
                [0, 1],
                [0],
                [1],
                [],
            ],
            # [1]
            [
                [0, 1],
                [0, 1],
                [1],
                [0, 1],
                [0],
                [1],
                [],
            ],
            # [sj]
            [
                [0],
                [0],
                [],
                [0],
                [0],
                [],
                [],
            ],
            # [1,1]
            [
                [0, 1],
                [0, 1],
                [1],
                [0, 1],
                [0],
                [1],
                [],
            ],
            # [si,1]
            [
                [1],
                [1],
                [1],
                [1],
                [],
                [1],
                [],
            ],
            # [1,sj]
            [
                [0],
                [0],
                [],
                [0],
                [0],
                [],
                [],
            ],
            # [si,sj]
            [
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ]
        ]

        for i, a in enumerate(ten_a):
            for j, b in enumerate(ten_b):
                ea = a.expand(*[si, sj])
                eb = b.expand(*[sj, si])
                self.helperTestReduceEquation(
                    eq, ea, eb, expected_equations[i][j],
                    expected_output_shape, expected_unsqueeze[i][j])

                self.helperTestReducedEinsum(
                    eq, [a, b], [[si, sj], [sj, si]])


    def helperTestReduceEquation(
            self,
            eq: str,
            ten_a: torch.Tensor,
            ten_b: torch.Tensor,
            expected_equation: str,
            expected_output_shape: List[int],
            expected_unsqueeze: List[int]):

        compiled_eq = compile_equation(eq)

        eeq = expected_equation
        eun = expected_unsqueeze
        ref_ceq = compile_equation(eeq)
        (shrunk_tensors, reduced_eq, unsqueeze_index,
         output_shape, factor_vars, factor) = reduce_equation(compiled_eq, [ten_a, ten_b])

        self.assertEqual(reduced_eq.input_variables, ref_ceq.input_variables, (ten_a, ten_b))
        self.assertEqual(reduced_eq.output_variables, ref_ceq.output_variables)
        self.assertEqual(output_shape, expected_output_shape)
        for st, dims in zip(shrunk_tensors, reduced_eq.input_variables):
            self.assertEqual(st.dim(), len(dims))
        self.assertEqual(unsqueeze_index, eun)


    def helperTestReducedEinsum(
            self,
            eq: str,
            tens: List[torch.Tensor],
            ten_expand_shapes: List[List[int]]):
        compiled_eq = compile_equation(eq)

        tens = [t.clone() for t in tens]
        expanded_tens = [t.expand(*s) for t, s in zip(tens, ten_expand_shapes)]
        (shrunk_tensors, reduced_eq, unsqueeze_index,
         output_shape, factor_vars, factor) = reduce_equation(compiled_eq, expanded_tens)

        expected_out = einsum(
            compiled_eq, *expanded_tens,
            block_size=AUTOMATIC_BLOCK_SIZE)
        actual_out = einsum(
            reduced_eq, *shrunk_tensors,
            block_size=AUTOMATIC_BLOCK_SIZE)
        semiring = RealSemiring(dtype=tens[0].dtype, device=tens[0].device)
        actual_out = post_einsum(actual_out, unsqueeze_index, output_shape, factor, semiring)
        self.assertTEqual(actual_out, expected_out)

        viterbi_expected_out, viterbi_expected_ptr = log_viterbi_einsum_forward(
            compiled_eq, *expanded_tens,
            block_size=AUTOMATIC_BLOCK_SIZE)
        viterbi_actual_out, viterbi_actual_ptr = log_viterbi_einsum_forward(
            reduced_eq, *shrunk_tensors,
            block_size=AUTOMATIC_BLOCK_SIZE)
        viterbi_semiring = ViterbiSemiring(dtype=tens[0].dtype, device=tens[0].device)
        viterbi_actual_out = post_einsum(
            viterbi_actual_out, unsqueeze_index, output_shape, factor,
            viterbi_semiring)
        viterbi_actual_ptr = post_einsum_argmax(
            viterbi_actual_ptr, unsqueeze_index, output_shape, factor_vars)
        self.assertTEqual(viterbi_actual_out, viterbi_expected_out)
        self.assertTEqual(viterbi_actual_ptr, viterbi_expected_ptr)


if __name__ == "__main__":
    unittest.main()
