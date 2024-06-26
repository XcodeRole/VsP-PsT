#pragma once

//#include <chrono>
//#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
//#define TOCK(x) std::cout << #x ": " << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count() << "s" << std::endl;

#include <tbb/tick_count.h>
#define TICK(x) auto bench_##x = tbb::tick_count::now();
#define TOCK(x) if(rank==0) Foam::Info << #x ": " << (tbb::tick_count::now() - bench_##x).seconds() << "s" << Foam::endl;

#define LINE Foam::Pout<< __FILE__ <<":"<<__LINE__<<Foam::endl;
