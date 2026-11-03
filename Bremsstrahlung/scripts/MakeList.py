from pathlib import Path

def MakeList(jobname, recostage_in, recostage_out, pressure, run_number):
    # Define the local directory path
    local_path = f"/ospool/ap40/data/krishan.mistry/job/Production/{jobname}/{pressure}/{recostage_in}_{run_number}/"

    # Get all files in the directory
    file_list = [f"{file.name}" for file in Path(local_path).iterdir() if file.is_file()]

    # Write to a text file
    output_file = f"../job/filelists/{jobname}_{recostage_out}_{run_number}.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(file_list))

    print(f"File list saved to {output_file}")

jobname="NEXT100_Th228_Port1a_FSPECT"


MakeList(jobname, "sophronia", "esmeralda", "4bar", "0")
MakeList(jobname, "sophronia", "esmeralda", "4bar", "15589")
MakeList(jobname, "sophronia", "esmeralda", "4bar", "15590")
