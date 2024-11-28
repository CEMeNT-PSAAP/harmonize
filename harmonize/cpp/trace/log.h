#ifndef HARMONIZE_TRACE_LOG
#define HARMONIZE_TRACE_LOG

#include "../util/iter.h"

namespace log
{



template<size_t SIZE>
class TinyLogEntryBase
{
    char const *format;
    char data[SIZE];
};


template<size_t SIZE, size_t ENTRY_COUNT>
class TinyLog
{

    AtomicIter<unsigned long long int> iter;
    TinyLogEntryBase<SIZE> entries;

    public:

    void clear()
    {
        iter.reset(0,ENTRY_COUNT);
    }

    template<typename... ARGS>
    bool log(char const *format, ARGS... args)
    {
        unsigned long long int index;
        if (iter.next(index)) {

        }
    };


};




} // namespace log

