#pragma once
#include <vector>

// Logical CPU ids in the order threads should be assigned:
//   1) first logical CPU of each SMT-capable core (P-cores on hybrid Intel)
//   2) each core without SMT (E-cores on hybrid Intel)
//   3) the remaining SMT sibling threads
// Only CPUs this process may run on are returned. Never empty.
std::vector<int> cpu_assignment_order();

// Prints the order and the sibling groups, for verification.
void print_cpu_topology();