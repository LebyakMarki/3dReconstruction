#include <mutex>
#include <deque>

#ifndef AKS_PROEKT_THREADSAFE_QUEUE_H
#define AKS_PROEKT_THREADSAFE_QUEUE_H

#endif //AKS_PROEKT_THREADSAFE_QUEUE_H


template<class T>
class ThreadSafeQueue{
private:
    mutable std::mutex m_m;
    std::condition_variable cv_m;
    std::deque<T> safe_queue;
    std::atomic<int> users{0};
    int queue_max_size = 40000;
public:
    ThreadSafeQueue()= default;

    void push_el(T new_el){
        std::unique_lock<std::mutex> lg{m_m};
        cv_m.wait(lg, [this](){ return safe_queue.size() < queue_max_size;});
        safe_queue.push_front(new_el);
        lg.unlock();
        cv_m.notify_one();
    }

    T wait_pop_el(){
        std::unique_lock<std::mutex> lg{m_m};
        cv_m.wait(lg, [this](){ return !safe_queue.empty();});
        T popped = safe_queue.back();
        safe_queue.pop_back();
        lg.unlock();
        if (safe_queue.size() > queue_max_size / 3){
            cv_m.notify_all();}
        else {
            cv_m.notify_one();}
        return popped;
    }

    void plus_user() {
        ++users;
    }

    void minus_user() {
        --users;
    }

    size_t users_amount() const {
        std::lock_guard<std::mutex> lg{m_m};
        return users;
    }
};


