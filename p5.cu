#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cmath>

int main() {
    char *S = "GAA"
    char *T = "GAA"
    N = strlen(S)
    M = strlen(T)

    device_vector<int> calc[2];
    calc[0].resize(N+1);  // linha anterior
    calc[1].resize(N+1);  // resultado temporário da transformação

    // preenche a linha anterior com zeros
    thrust::fill(calc[0].begin(), calc[0].end(),0);

    // copia a seq S pra d_S
    device_vector<char> d_S();
    thrust::copy(S, S+N, d_S.begin());

    for (int j=0; j<M; j++)
    {
        char letradeT = T[j];
        thrust::transform(c0, c1, calc[1].begin()+1, meu_functor(d_S.data(), letradeT, calc[0].data()));
        thrust::inclusive_scan(calc[1].begin+1, calc[1].end(), calc[0].begin()+1, thrust::maximum<int>());
    }

}


// Tratar tamanhos diferentes no functor
// c1 e c2 são counting iterators pra resolver o problema de que precisa ter 2 vectors de tamanhos =s


// functor pode receber device_ptr como parametro para apontar o inicio do device_vector

// diagonal e superior vem de calc[0]
// comparar com letradeT pra ver se é match ou mismatch

// na linha do functor
// posicao de S chamada de j
// para usar declarar: operator() int(&j) const