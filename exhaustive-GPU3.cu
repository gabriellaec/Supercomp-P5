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
#include <vector>
#include <string>
#include<bits/stdc++.h>
#include <omp.h>

using namespace std;

// ----- Pesos definidos ----- //
#define WMAT 2
#define WMIS -1
#define WGAP -1

// ----- Structs ----- //

struct combination {
    long value;
    vector<char> seq_a;
    vector<char> seq_b;
};

// ----- Functors ----- //

// Functor para o cálculo do score na GPU
struct meu_functor
{
    thrust::device_ptr<char> d_S;
    thrust::device_ptr<int> calc0;
    char letradeT;

    meu_functor( thrust::device_ptr<char> d_S_, char letradeT_, thrust::device_ptr<int>  calc0_) : d_S(d_S_), letradeT(letradeT_), calc0(calc0_) {};
    __host__ __device__
    int operator() (const int(&j) ){
        int diagonal, insercao;

        insercao = calc0[j] - 1;

        if (d_S[j] == letradeT) {
            diagonal = calc0[j-1] + WMAT;
        } else {
            diagonal = calc0[j-1] + WMIS;
        }

        int max = 0;
        if (diagonal > max) {
            max = diagonal;
        }

        if (insercao > max) {
            max = insercao;
        }

        return max;
    }
};




// ----- Funções ----- //

// Função que calcula os scores de todas as combinações do batch passado para a GPU 
// e retorna o melhor deles 
int gpu_score(vector<vector<char>> subseqs_a, vector<vector<char>> subseqs_b, int global_max){

    int maior = 0;
    int max_possible=0;
     for (auto& S : subseqs_a){
      for (auto& T : subseqs_b){
        int N = S.size();
        int M = T.size();

        if (N>M) max_possible = M*2;
        else max_possible = N*2;
        if (maior<max_possible){


        thrust::device_vector<int> calc[2]; // precisa ser um vector
        calc[0].resize(N+1);  // linha anterior
        calc[1].resize(N+1);  // resultado temporário da transformação

        // preenche a linha anterior com zeros
        thrust::fill(calc[0].begin(), calc[0].end(),0);

        // copia a seq S pra d_S
        thrust::device_vector<char> d_S(N);
        thrust::copy( S.begin(), S.begin()+N, d_S.begin());

        // counting iterators
        thrust::counting_iterator<int> c0(1);
        thrust::counting_iterator<int> c1(M+1);

        for (int j=0; j<M; j++)
        {
            char letradeT = T[j];
            thrust::transform(c0, c1, calc[1].begin()+1, meu_functor(d_S.data(), letradeT, calc[0].data()));  // calcula a diagonal e superior
            thrust::inclusive_scan(calc[1].begin()+1, calc[1].end(), calc[0].begin()+1, thrust::maximum<int>());  // calcula o lateral
        }

        // int max = thrust::reduce(calc[1].begin()+1, calc[1].end(),-1,thrust::maximum<int>());
        int max = calc[1].data()[calc[1].size()-1];

        if (max>maior) maior = max;

        if (max >= global_max) return max;

        }

    }
    }

    return maior;
}

// Fução usada para gerar as subsequências
void gera_subseq(string seq, int start_point, int end_point, vector<vector<char>>& matriz_subseq){
    if (end_point > (int)seq.size())
      return;
    else if (start_point > end_point){
        gera_subseq(seq, 0, end_point+1, matriz_subseq);
    }else{
        if (start_point != end_point){
            vector<char> subseq;
            for (int j=start_point; j<end_point; j++){
                subseq.push_back(seq[j]);
            }
            matriz_subseq.push_back(subseq);
        }
        gera_subseq(seq, start_point+1, end_point, matriz_subseq);
    }
}

int main() {
    double init_time, final_time;
    init_time = omp_get_wtime();

// Lendo as sequências de um arquivo
    int N,M;
    string base;

    cin >> N >> M;
    
    string S_str;
    string T_str;

    cin >> S_str;
    cin >> T_str;

// Gerando as subsequências
    vector<vector<char>> subseqs_a;
    vector<vector<char>> subseqs_b;
    gera_subseq((S_str),0,1, subseqs_a);
    gera_subseq((T_str),0,1,subseqs_b);

    vector<char> T(T_str.begin(), T_str.end());
    vector<char> S(S_str.begin(), S_str.end());

// Calculando os scores
    int melhor_valor_g = -1;  // melhor valor global
    int global_max = 0;

    if (T.size() > S.size()) global_max = 2*S.size();
    else global_max = 2*T.size();

    melhor_valor_g = gpu_score(subseqs_a,subseqs_b, global_max);

    // cout << endl << "result: " << melhor_valor_g << endl;
    // final_time = omp_get_wtime() - init_time;
    // cout << "tempo: " << final_time << endl;

    cout << melhor_valor_g << endl;

    return 0;

}
