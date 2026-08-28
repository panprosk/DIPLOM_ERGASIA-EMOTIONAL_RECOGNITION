from data.deap_loader import DEAPLoader


loader = DEAPLoader("data/deap")

dataset = loader.load_all_subjects()

print()

print("Subjects:", len(dataset))

print()

for subject_name in dataset:

    print(subject_name)

    print(dataset[subject_name]["data"].shape)

    print(dataset[subject_name]["labels"].shape)

    print()

    break