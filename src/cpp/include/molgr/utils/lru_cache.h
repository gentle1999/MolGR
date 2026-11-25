#pragma once

#include <cstddef>
#include <list>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace molgr
{
    namespace utils
    {
        /// Size-limited string-keyed cache that updates recency on successful lookup.
        /// @tparam Value Payload type stored for each molecular key.
        template <typename Value>
        class StringLruCache
        {
        public:
            /// Set the maximum number of entries retained before evicting the least recently used key.
            /// @param max_size Maximum entries retained in the least-recently-used cache.
            explicit StringLruCache(std::size_t max_size)
                : max_size_(max_size)
            {
            }

            /// @param key Lookup key for the recorded state or property.
            /// @param value Output payload, written only on a cache hit.
            /// @return True on a cache hit.
            bool Get(const std::string &key, Value &value)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = index_.find(key);
                if (it == index_.end())
                {
                    ++misses_;
                    return false;
                }
                items_.splice(items_.begin(), items_, it->second);
                value = it->second->value;
                ++hits_;
                return true;
            }

            /// Store the value as most recently used, evicting an old entry if necessary.
            /// @param key Lookup key for the recorded state or property.
            /// @param value Payload to retain for this key.
            void Put(std::string key, Value value)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = index_.find(key);
                if (it != index_.end())
                {
                    it->second->value = std::move(value);
                    items_.splice(items_.begin(), items_, it->second);
                    return;
                }

                items_.push_front(Entry{std::move(key), std::move(value)});
                index_[items_.front().key] = items_.begin();
                while (index_.size() > max_size_)
                {
                    index_.erase(items_.back().key);
                    items_.pop_back();
                }
            }

            /// @return Hits, misses and current size in that order.
            std::tuple<std::size_t, std::size_t, std::size_t> Info() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return {hits_, misses_, items_.size()};
            }

            /// Discard entries and reset the cache statistics.
            void Clear()
            {
                std::lock_guard<std::mutex> lock(mutex_);
                items_.clear();
                index_.clear();
                hits_ = 0;
                misses_ = 0;
            }

        private:
            /// Key and cached payload stored together in recency order.
            struct Entry
            {
                std::string key;
                Value value;
            };

            std::size_t max_size_;
            mutable std::mutex mutex_;
            std::list<Entry> items_;
            std::unordered_map<std::string, typename std::list<Entry>::iterator> index_;
            std::size_t hits_ = 0;
            std::size_t misses_ = 0;
        };
    }
}
