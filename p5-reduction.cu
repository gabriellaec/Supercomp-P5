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
struct item {
    long item_score;
    vector<char> seq_a;
    vector<char> seq_b;
};

struct combination {
    long value;
    vector<char> seq_a;
    vector<char> seq_b;
};



// ----- Functors ----- //
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


int gpu_score(vector<char> S, vector<char> T, int N, int M){
    thrust::device_vector<int> calc[2]; // precisa ser um vector
    calc[0].resize(N+1);  // linha anterior
    calc[1].resize(N+1);  // resultado temporário da transformação

    // preenche a linha anterior com zeros
    thrust::fill(calc[0].begin(), calc[0].end(),0);

    // copia a seq S pra d_S
    thrust::device_vector<char> d_S(N);
    thrust::copy( S.begin(), S.begin()+N, d_S.begin());

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

    int max = calc[1].data()[calc[1].size()-1];

    return max;
}



// void gera_subseq(string seq, long start_point, long end_point, vector<char>& subseqs){
//     if (end_point > (long)seq.size())
//       return;
//     else if (start_point > end_point){
//         gera_subseq(seq, 0, end_point+1, subseqs);
//     }else{
//         if (start_point != end_point){
//             string subseq;
//             for (long j=start_point; j<end_point; j++){
//                 subseq.push_back(seq[j]);
//             }
//             subseqs.push_back(subseq);
//         }
//         gera_subseq(seq, start_point+1, end_point, subseqs);
//     }
// }


void gera_subseq(string input, vector<vector<char>> &all){
    for (int j=0; j<(int)input.size(); j++){
        for(int i=0; i<(int)input.size(); i++){
            string sub = input.substr(i,j);
            vector<char> sub_char(sub.begin(), sub.end());
            if (!(sub_char.size()) < 1)
            all.push_back(sub_char);
        }
    }
}



int main() {
    int N,M;
    string base;

    cin >> N >> M;
    string S_str;
    string T_str;

    cin >> S_str;
    cin >> T_str;
    
    // char *S;
    // char *T;

    // cin >> base;
    // for(int i = 0; i < N; i++){
    //     S.push_back(base[i]);
    // }
    // cin >> base;
    // for(int i = 0; i < M; i++){
    //     T.push_back(base[i]);
    // }

    // int N,M;
    // char *S = "AGCA";
    // char *T = "ACACA";
    // N = strlen(S);
    // M = strlen(T);


    vector<vector<char>> subseqs_a;
    vector<vector<char>> subseqs_b;
    gera_subseq(S_str, subseqs_a);
    gera_subseq(T_str,subseqs_b);


    
    // strcpy(T, T_str);

    vector<char> T(T_str.begin(), T_str.end());
    vector<char> S(S_str.begin(), S_str.end());

///////////////////////////////////////

    item melhor, sw_atual;    
    vector<combination> combinations((long)subseqs_a.size()*(long)subseqs_b.size());;  


    // cout << (long)subseqs_a.size()*(long)subseqs_b.size() << " combinations" << endl;
 
    long i=0;
    int melhor_valor = -1;
    int melhor_valor_g = -1;
    int val;

    for (auto& sub_a : subseqs_a){
        for (auto& sub_b : subseqs_b){
            combinations.push_back({i,sub_a, sub_b});
            i+=1;
            if (i>=600000){ // divisão em sub blocos para não estourar o vetor
                #pragma omp parallel for reduction(max:melhor_valor)
                for (auto& el : combinations){ 
                    val = gpu_score(el.seq_a, el.seq_b,(el.seq_a).size(),(el.seq_b).size());
                    if (val>melhor_valor)melhor_valor=val;
                } 

                for (int i=0; i<(int)resultados.size(); i++){
                    if (melhor_valor > melhor_valor_g){
                        melhor_valor_g = melhor_valor;
                    }
                }
            // -----------------------------------//
            i = 0;   
            combinations.clear();             
            }
        }
    }


    #pragma omp parallel for reduction(max:melhor_valor)
    for (auto& el : combinations){ 
            val = gpu_score(el.seq_a, el.seq_b,(el.seq_a).size(),(el.seq_b).size());
            if (val>melhor_valor)melhor_valor=val;
            
    }
    
        
    for (int i=0; i<(int)resultados.size(); i++){
        if (melhor_valor > melhor_valor_g){
            melhor_valor_g = melhor_valor;
        }
    }

//////////////////////////////////////
    // vector<int> resultados((long)combinations.size());

    // cout << "--------------------";
    // #pragma omp parallel for shared(resultados) 
    // for (auto& el : combinations){
    //     resultados[el.value] = gpu_score(el.seq_a, el.seq_b,(el.seq_a).size(),(el.seq_b).size());
    // }
//////////////////////////////////////


    // long melhor_valor=-1;
    // for (long i=0; i<(long)resultados.size(); i++){
    //     if (resultados[i] > melhor_valor){
    //         melhor_valor = resultados[i];
    //     }
    // }



    cout << "maximo: " << melhor_valor_g << endl;

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