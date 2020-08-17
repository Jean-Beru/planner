from __future__ import print_function
from ortools.sat.python import cp_model
import sys
import json
import math

class Data:
    def __init__(self, filename):
        json_data = json.load(open(filename))
        self.days = json_data['days']
        self.shifts = json_data['shifts']
        self.users = json_data['users']
        self.wishes = json_data['wishes']
    def GetUsersRange(self):
        return range(len(self.users))
    def GetDaysRange(self):
        return range(len(self.days))
    def GetShiftsRange(self):
        return range(len(self.shifts))

def main():
    # Parse arguments
    if len(sys.argv) != 2:
        print('usage: %s data.json' % sys.argv[0])
        exit(1)
    file = sys.argv[1]

    # Populate data
    data = Data(file)
    users_range = data.GetUsersRange()
    days_range = data.GetDaysRange()
    shifts_range = data.GetShiftsRange()

    # Creates the model.
    model = cp_model.CpModel()

    # Creates shift variables.
    shifts = {}
    for u in users_range:
        for d in days_range:
            for s in shifts_range:
                shifts[(u, d, s)] = model.NewBoolVar('shift_u%id%is%i' % (u, d, s))

    # Each shift is assigned to exactly 1 user in the schedule period.
    for d in days_range:
        for s in shifts_range:
            model.Add(sum(shifts[(u, d, s)] for u in users_range) == 1)

    # Each user works at most one shift per day.
    for u in users_range:
        for d in days_range:
            model.Add(sum(shifts[(u, d, s)] for s in shifts_range) <= 1)

    # Each user can have constraints
    users_presence_ratio = {}
    for u in users_range:
        users_constraint_count = 0
        for d in days_range:
            for s in shifts_range:
                if data.wishes[u][d][s] == 0:
                    model.Add(shifts[(u, d, s)] == 0)
                    users_constraint_count = users_constraint_count + 1
        users_presence_ratio[u] = (100 - ((users_constraint_count * 100 ) // (len(days_range) * len(shifts_range)))) * .01
    model.Maximize(sum(data.wishes[u][d][s] * shifts[(u, d, s)] for u in users_range for d in days_range for s in shifts_range))

    # min_shifts_per_user is the largest integer such that every user can be assigned at least that many shifts.
    # If the number of users doesn't divide the total number of shifts over the schedule period, some users have to work
    # one more shift, for a total of min_shifts_per_user + 1.
    # a user coefficient is applied to min_shifts_per_user and max_shifts_per_user in order to remove not-counted absences like holidays
    min_shifts_per_user = (len(shifts_range) * len(days_range)) // len(users_range)
    max_shifts_per_user = min_shifts_per_user + 1
    for u in users_range:
        num_shifts_worked = sum(shifts[(u, d, s)] for d in days_range for s in shifts_range)
        model.Add(math.floor(min_shifts_per_user * users_presence_ratio[u]) <= num_shifts_worked)
        model.Add(num_shifts_worked <= math.ceil(max_shifts_per_user * users_presence_ratio[u]))

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print result
    if (status == cp_model.OPTIMAL):
        for d in days_range:
            print(data.days[d])
            for s in shifts_range:
                for u in users_range:
                    if solver.Value(shifts[(u, d, s)]) == 1:
                        print('  %-15s %s' % (data.shifts[s], data.users[u]))
    else:
        print('No solution found.')

    # Statistics.
    print()
    print('Statistics')
    print('  - score      : %i' % solver.ObjectiveValue())
    print('  - conflicts  : %i' % solver.NumConflicts())
    print('  - branches   : %i' % solver.NumBranches())
    print('  - wall time  : %f s' % solver.WallTime())

if __name__ == '__main__':
    main()
