from __future__ import print_function
from ortools.sat.python import cp_model
import json

class Data:
    def __init__(self, filename):
        json_data = json.load(open(filename))
        self.users = json_data['users']
        self.days = json_data['days']
        self.shifts = json_data['shifts']
        self.blocks = json_data['blocks']
        self.wishes = json_data['wishes']
    def GetUsersRange(self):
        return range(len(self.users))
    def GetDaysRange(self):
        return range(len(self.days))
    def GetShiftsRange(self):
        return range(len(self.shifts))

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, shifts, data, sols):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.shifts = shifts
        self.data = data
        self.solutions = set(sols)
        self.solution_count = 0

    def OnSolutionCallback(self):
        if self.solution_count in self.solutions:
            print('Solution %i' % self.solution_count)
            for d, day in enumerate(self.data.days):
                print(day)
                for s, shift in enumerate(self.data.shifts):
                    for u, user in enumerate(self.data.users):
                        if self.Value(self.shifts[(u, d, s)]):
                            print('  %-15s %s' % (shift, user))

            print('Repartition %i' % self.solution_count)
            days_range = self.data.GetDaysRange()
            shifts_range = self.data.GetShiftsRange()
            for u, user in enumerate(self.data.users):
                user_shift_count = sum(self.Value(self.shifts[(u, d, s)]) for d in days_range for s in shifts_range)
                print('  %-15s %d' % (user, user_shift_count))

            print()
        self.solution_count += 1
        if self.solution_count >= len(self.solutions):
            self.StopSearch()

    def GetSolutionCount(self):
        return self.solution_count

def main():
    # Populate data
    data = Data('./data.json')
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

    # Each user can have blocking constraints.
    for b, block in enumerate(data.blocks):
        model.Add(shifts[(block[0], block[1], block[2])] == 0)

    # Each use can have wishes
    wishes = {}
    for u in users_range:
        for d in days_range:
            for s in shifts_range:
                wishes[(u, d, s)] = 1
    for w in data.wishes:
        wishes[(w[0], w[1], w[2])] = 0
    model.Maximize(sum(wishes[(u, d, s)] * shifts[(u, d, s)] for u in users_range for d in days_range for s in shifts_range))

    # min_shifts_per_user is the largest integer such that every user can be assigned at least that many shifts.
    # If the number of users doesn't divide the total number of shifts over the schedule period, some users have to work
    # one more shift, for a total of min_shifts_per_user + 1.
    min_shifts_per_user = (len(shifts_range) * len(days_range)) // len(users_range)
    max_shifts_per_user = min_shifts_per_user + 1
    for u in users_range:
        num_shifts_worked = sum(shifts[(u, d, s)] for d in days_range for s in shifts_range)
        model.Add(min_shifts_per_user <= num_shifts_worked)
        model.Add(num_shifts_worked <= max_shifts_per_user)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.Solve(model)

    # Print result
    for d, day in enumerate(data.days):
        print(day)
        for s, shift in enumerate(data.shifts):
            for u, user in enumerate(data.users):
                if solver.Value(shifts[(u, d, s)]) == 1:
                    if wishes[(u, d, s)] == 1:
                        print('  %-15s %s' % (shift, user))
                    else:
                        print('  %-15s %s (unwanted)' % (shift, user))

    # Statistics.
    print()
    print('Statistics')
    print('  - score      : %i' % solver.ObjectiveValue())
    print('  - conflicts  : %i' % solver.NumConflicts())
    print('  - branches   : %i' % solver.NumBranches())
    print('  - wall time  : %f s' % solver.WallTime())

if __name__ == '__main__':
    main()
