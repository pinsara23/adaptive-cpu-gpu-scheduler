#include "cpu_topology.hpp"
#include <sched.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

// Parses sysfs lists such as "0,1", "0-1" or "12".
std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        size_t dash = token.find('-');
        if (dash == std::string::npos) {
            out.push_back(std::stoi(token));
        } else {
            int a = std::stoi(token.substr(0, dash));
            int b = std::stoi(token.substr(dash + 1));
            for (int c = a; c <= b; ++c) out.push_back(c);
        }
    }
    return out;
}

std::vector<int> siblings_of(int cpu) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                    "/topology/thread_siblings_list");
    std::string line;
    if (!f.is_open() || !std::getline(f, line)) return {cpu};
    std::vector<int> v = parse_cpu_list(line);
    if (v.empty()) v.push_back(cpu);
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

std::vector<int> cpu_assignment_order() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    sched_getaffinity(0, sizeof(allowed), &allowed);

    std::vector<int> smt_primaries, single_cores, secondaries;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        std::vector<int> sib = siblings_of(cpu);
        if (cpu == sib.front()) {
            (sib.size() > 1 ? smt_primaries : single_cores).push_back(cpu);
        } else {
            secondaries.push_back(cpu);
        }
    }

    std::vector<int> order;
    order.insert(order.end(), smt_primaries.begin(), smt_primaries.end());
    order.insert(order.end(), single_cores.begin(), single_cores.end());
    order.insert(order.end(), secondaries.begin(), secondaries.end());
    if (order.empty()) order.push_back(0);
    return order;
}

void print_cpu_topology() {
    std::vector<int> order = cpu_assignment_order();
    std::cout << "Assignment order (" << order.size() << " CPUs):";
    for (int c : order) std::cout << " " << c;
    std::cout << "\n";
    for (int c : order) {
        std::vector<int> sib = siblings_of(c);
        if (c != sib.front()) continue;
        std::cout << "  core group:";
        for (int s : sib) std::cout << " " << s;
        std::cout << (sib.size() > 1 ? "  (SMT)" : "  (single thread)") << "\n";
    }
}