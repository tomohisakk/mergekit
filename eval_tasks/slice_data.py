import datasets

ds = datasets.load_dataset("elyza/ELYZA-tasks-100")
slice_ds = ds["test"].select(range(10))

ds["test"] = slice_ds

ds.save_to_disk("./slice_et100_10")
