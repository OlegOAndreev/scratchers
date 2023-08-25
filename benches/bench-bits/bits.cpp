#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <ratio>
#include <string>
#include <system_error>
#include <vector>

void array_partition(const std::vector<std::string>& strings, std::vector<int>& partitions)
{
    bool mask[256];
    for (size_t i = 0; i < strings.size(); i++) {
        std::fill(mask, mask + 256, false);
        int ret = 1;
        for (char b : strings[i]) {
            if (mask[b]) {
                std::fill(mask, mask + 256, false);
                ret++;
            }
            mask[b] = true;
        }
        partitions[i] = ret;
    }
}

void bitmask_partition(const std::vector<std::string>& strings, std::vector<int>& partitions)
{
    for (size_t i = 0; i < strings.size(); i++) {
        size_t mask = 0;
        int ret = 1;
        for (char b : strings[i]) {
            size_t m = size_t(1) << (b - 'A');
            if (mask & m) {
                mask = m;
                ret++;
            } else {
                mask |= m;
            }
        }
        partitions[i] = ret;
    }
}

void array_noclear_partition(const std::vector<std::string>& strings, std::vector<int>& partitions)
{
    int prev[256];
    for (size_t i = 0; i < strings.size(); i++) {
        std::fill(prev, prev + 256, 0);
        int ret = 1;
        for (char b : strings[i]) {
            if (prev[b] == ret) {
                ret++;
            }
            prev[b] = ret;
        }
        partitions[i] = ret;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <data file> <algo>\n", argv[0]);
        return 1;
    }
    std::string filename(argv[1]);
    std::string algo(argv[2]);

    auto start_load = std::chrono::high_resolution_clock::now();
    std::vector<std::string> strings;
    std::vector<int> partitions;
    std::ifstream f(filename);
    std::string line;
    while (std::getline(f, line)) {
        size_t pos = line.find(' ');
        if (pos != std::string::npos) {
            std::string s(line, 0, pos);
            strings.push_back(std::move(s));
            int p;
            if (auto [dummy, ec] = std::from_chars(line.data() + pos + 1, line.data() + line.size(), p); ec != std::errc()) {
                printf("Error parsing %s: %s\n", line.c_str(), std::make_error_code(ec).message().c_str());
                return 1;
            }
            partitions.push_back(p);
        }
    }
    auto load_time = std::chrono::high_resolution_clock::now() - start_load;
    printf("Parsed data in %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(load_time).count());

    std::vector<int> our_partitions(partitions.size());
    auto start_compute = std::chrono::high_resolution_clock::now();

    if (algo == "array") {
        array_partition(strings, our_partitions);
    } else if (algo == "bitmask") {
        bitmask_partition(strings, our_partitions);
    } else if (algo == "array_noclear") {
        array_noclear_partition(strings, our_partitions);
    } else {
        printf("Unknown algo %s\n", algo.c_str());
        return 1;
    }
    auto compute_time = std::chrono::high_resolution_clock::now() - start_compute;
    for (size_t i = 0; i < partitions.size(); i++) {
        if (partitions[i] != our_partitions[i]) {
            printf("Different partitions for %s (line %d): %d vs %d\n", strings[i].c_str(), int(i), partitions[i], our_partitions[i]);
            return 1;
        }
    }
    printf("Computed using algo %s in %lldms\n", algo.c_str(),
        std::chrono::duration_cast<std::chrono::milliseconds>(compute_time).count());

    return 0;
}
