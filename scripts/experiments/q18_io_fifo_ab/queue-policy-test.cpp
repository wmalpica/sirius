#include "blockingconcurrentqueue.h"
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

void test_preference(bool fifo) {
  duckdb_moodycamel::BlockingConcurrentQueue<int> q;
  duckdb_moodycamel::ProducerToken producer(q);
  std::mutex mutex;
  auto enqueue = [&](int id) { std::lock_guard lock(mutex); assert(fifo ? q.enqueue(producer,id) : q.enqueue(id)); };
  std::atomic<int> turn{0}, refill{-1}, ack{0}; std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  for(int id=0;id<8;++id) threads.emplace_back([&,id]{
    while(turn.load()!=id) std::this_thread::yield();
    enqueue(id);turn.fetch_add(1);
    while(!stop.load()) {
      if(refill.load()==id){enqueue(id);refill.store(-1);ack.fetch_add(1);}
      else std::this_thread::yield();
    }
  });
  while(turn.load()!=8) std::this_thread::yield();
  int counts[8]={};
  for(int i=0;i<800;++i){
    int x=-1;assert(q.try_dequeue(x));++counts[x];
    if(fifo) assert(x==i%8);
    int prev=ack.load();refill.store(x);while(ack.load()==prev)std::this_thread::yield();
  }
  stop.store(true);for(auto &t:threads)t.join();
  std::cout<<(fifo?"FIFO":"original");for(int x:counts)std::cout<<" "<<x;std::cout<<"\n";
}
void test_concurrent_fifo() {
  duckdb_moodycamel::BlockingConcurrentQueue<std::unique_ptr<int>> q;
  duckdb_moodycamel::ProducerToken producer(q);
  std::mutex mutex; int sequence=0;
  std::vector<std::thread> threads;
  for(int i=0;i<8;++i) threads.emplace_back([&]{
    for(int j=0;j<10000;++j){
      std::lock_guard lock(mutex);
      assert(q.enqueue(producer,std::make_unique<int>(sequence++)));
    }
  });
  for(int n=0;n<80000;++n){
    std::unique_ptr<int> value;
    assert(q.wait_dequeue_timed(value,5000000));assert(*value==n);
  }
  for(auto &t:threads)t.join();
  std::unique_ptr<int> value;assert(!q.try_dequeue(value));
  std::cout<<"FIFO: 80000 concurrent move-only requests preserve enqueue order, no loss/duplicates\n";
}
int main(){test_preference(false);test_preference(true);test_concurrent_fifo();}
