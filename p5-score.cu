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
#include <thrust/transform.h>

using namespace std;

// ----- Pesos definidos ----- //
#define WMAT 2
#define WMIS -1
#define WGAP -1


struct meu_functor
{
    thrust::device_ptr<char> d_S;
    thrust::device_ptr<int> calc0;
    char letradeT;

    meu_functor( thrust::device_ptr<char> d_S_, char letradeT_, thrust::device_ptr<int>  calc0_) : d_S(d_S_), letradeT(letradeT_), calc0(calc0_) {};
    __host__ __device__
    int operator() (const int(&j) ){

        if (d_S[j] == letradeT)
            return calc0[j]+WMAT;
        else if (d_S[j] != letradeT)
            return calc0[j]+WMIS;  // mismatch
        else
            return calc0[j]+WGAP;
           
    }
};



int main() {

    int N,M;
    string base;

    cin >> N >> M;
    char *S;
    char *T;

    cin >> S;
    cin >> T;


    // int N,M;
    // char base;

    // cin >> N >> M;
    // char *S;
    // char *T;
    
    // // vector<char> S(N+1,0);
    // // vector<char> T(M,0);

    // cin >> S;
    // // for(int i = 0; i < N; i++){
    // //     S.push_back(base[i]);
    // // }
    // cin >> T;
    // for(int i = 0; i < M; i++){
    //     T.push_back(base[i]);
    // }

    // int N,M;
    // char *S = "AGCA";
    // char *T = "ACACA";
    // N = strlen(S);
    // M = strlen(T);

    thrust::device_vector<int> calc[2]; // precisa ser um vector
    calc[0].resize(N+1);  // linha anterior
    calc[1].resize(N+1);  // resultado temporário da transformação

    // preenche a linha anterior com zeros
    thrust::fill(calc[0].begin(), calc[0].end(),0);

    // copia a seq S pra d_S
    thrust::device_vector<char> d_S(N);
    thrust::copy(S, S+N, d_S.begin());

    thrust::counting_iterator<int> c0(1);
    thrust::counting_iterator<int> c1(M+1);

    // std::cout << d_S.data();

    for (int j=0; j<M; j++)
    {
        char letradeT = T[j];
        thrust::transform(c0, c1, calc[1].begin()+1, meu_functor(d_S.data(), letradeT, calc[0].data()));
        thrust::inclusive_scan(calc[1].begin()+1, calc[1].end(), calc[0].begin()+1, thrust::maximum<int>());
    }


    // for (int i=0; i< N; i++)
    //      cout << "maximo: " << el << endl;

    int max = thrust::reduce(calc[1].begin(), calc[1].end(), thrust::maximum<int>());
    cout << "maximo: " << max;

    return 0;

}


// Tratar tamanhos diferentes no functor
// c1 e c2 são counting iterators pra resolver o problema de que precisa ter 2 vectors de tamanhos =s


// functor pode receber device_ptr como parametro para apontar o inicio do device_vector

// diagonal e superior vem de calc[0]
// comparar com letradeT pra ver se é match ou mismatch

// na linha do functor
// posicao de S chamada de j
// para usar declarar: operator() int(&j) const


// nvcc -arch=sm_70 -std=c++14 p5.cu -o p5