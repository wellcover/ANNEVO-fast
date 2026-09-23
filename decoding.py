import argparse
from src.gene_decoding import gene_structure_decoding
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Decode gene structure based on deep learning model's prediction.")
    parser.add_argument("-g", "--genome", required=True, help="Genome to be decoded.")
    parser.add_argument("-p", "--model_prediction_path", required=True,
                        help="Path to the probability predicted by the model.")
    parser.add_argument("-o", "--output", required=True, help="Output GFF file")
    parser.add_argument("-t", "--threads", type=int, default=48, help="Number of CPU cores used for decoding.")
    parser.add_argument("--show_log", action="store_true", help="Show decoding progress bars.")

    parser.add_argument("--min_intron_length", type=int, default=20,
                        help="Minimum intron length of CDS-associated intron groups.")
    parser.add_argument("--min_prot_length", type=int, default=100,
                        help="Predicted proteins shorter than this length are filtered with a higher confidence threshold.")
    parser.add_argument("--gcag_penalty", type=float, default=10.0,
                        help="Extra log penalty for GC-AG donor sites relative to canonical GT (default 10).")
    parser.add_argument("--atac_penalty", type=float, default=None,
                        help="Enable AT-AC (U12-type) splice path with this extra log penalty; omit to disable (default).")
    parser.add_argument("--ave_threshold", type=float, default=0.1,
                        help="Window-mean genic probability threshold for candidate gene regions (default 0.1).")
    parser.add_argument("--max_threshold", type=float, default=0.5,
                        help="Per-base genic probability threshold; regions need >=min_cds_length bases above it (default 0.5).")
    parser.add_argument("--min_cds_score", type=float, default=0.6,
                        help="Mean CDS emission score filter; single-exon/short genes face 1.5x of it "
                             "(default 0.6, empirically calibrated for +0.2~0.4pp gene F1).")
    args = parser.parse_args()
    AVE_THRESHOLD = args.ave_threshold
    MAX_THRESHOLD = args.max_threshold
    MIN_CDS_LENGTH = args.min_prot_length * 3
    MIN_CDS_SCORE = args.min_cds_score
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    gene_structure_decoding(args.genome, args.model_prediction_path, args.output, args.threads,
                            AVE_THRESHOLD, MAX_THRESHOLD, MIN_CDS_LENGTH, MIN_CDS_SCORE, args.min_intron_length,
                            show_log=args.show_log,
                            gcag_penalty_extra=args.gcag_penalty, atac_extra=args.atac_penalty)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene decoding took {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main()
