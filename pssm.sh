#!/bin/sh

for file in $(find . -iname '*.fasta'); do
    filename=$(basename "$file")
    output_filename="./pssm/"$filename".txt"
    psiblast -subject "$file" -in_msa "$file" -out_ascii_pssm "$output_filename" -num_iterations 3
done
