import os

def main():
    report_file = "report.tex"
    if not os.path.exists(report_file):
        print(f"Error: {report_file} not found.")
        return

    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Abstract
    target_abstract = (
        "We implement a centroid-distance outlier removal method to clean label noise and a "
        "landmark-based visibility filtering system to eliminate false predictions on occluded body parts."
    )
    replacement_abstract = (
        "We implement a landmark-based visibility filtering system to eliminate false predictions "
        "on occluded body parts."
    )
    
    # 2. Project Objectives
    target_objectives = """                    To address the problem statement, this project is structured around six key technical objectives:
                    \\begin{itemize}
                        \\item \\textbf{O1 (Scale-Invariant Cropping):} Build a pose-driven cropping engine using MediaPipe to dynamically define bounding boxes for the head, torso, and hands, scaling box dimensions proportionally based on the worker's shoulder width.
                        \\item \\textbf{O2 (High-Dimensional Semantic Encoding):} Implement transfer learning by leveraging a pre-trained, headless ResNet18 model to extract 512-dimensional semantic feature vectors from cropped regions.
                        \\item \\textbf{O3 (Methodological Core Classification):} Design, train, and evaluate three binary Random Forest classifiers as the primary learning algorithms, optimizing their ensemble parameters.
                        \\item \\textbf{O4 (Centroid-Distance Outlier Pruning):} Develop a multivariate statistical outlier removal pipeline to detect and discard noisy or mislabeled bounding box crops from the training dataset.
                        \\item \\textbf{O5 (Landmark Visibility Enforcement):} Integrate MediaPipe's joint visibility and presence metrics into the cropping engine to dynamically suppress predictions on occluded body parts, outputting explicit \\texttt{NOT\\_VISIBLE} labels.
                        \\item \\textbf{O6 (Real-World Validation):} Deploy the pipeline on unseen real-world images, evaluate the generalization gap, and provide solutions to mitigate background context bias and resolution loss.
                    \\end{itemize}"""
                    
    replacement_objectives = """                    To address the problem statement, this project is structured around five key technical objectives:
                    \\begin{itemize}
                        \\item \\textbf{O1 (Scale-Invariant Cropping):} Build a pose-driven cropping engine using MediaPipe to dynamically define bounding boxes for the head, torso, and hands, scaling box dimensions proportionally based on the worker's shoulder width.
                        \\item \\textbf{O2 (High-Dimensional Semantic Encoding):} Implement transfer learning by leveraging a pre-trained, headless ResNet18 model to extract 512-dimensional semantic feature vectors from cropped regions.
                        \\item \\textbf{O3 (Methodological Core Classification):} Design, train, and evaluate three binary Random Forest classifiers as the primary learning algorithms, optimizing their ensemble parameters.
                        \\item \\textbf{O4 (Landmark Visibility Enforcement):} Integrate MediaPipe's joint visibility and presence metrics into the cropping engine to dynamically suppress predictions on occluded body parts, outputting explicit \\texttt{NOT\\_VISIBLE} labels.
                        \\item \\textbf{O5 (Real-World Validation):} Deploy the pipeline on unseen real-world images, evaluate the generalization gap, and provide solutions to mitigate background context bias and resolution loss.
                    \\end{itemize}"""

    # 3. Table 3 (Outlier Pruning Statistics) and class counts
    target_table = """                    A total of 159,586 individual object crops are extracted from the dataset folders. To clean this dataset, we apply the centroid-distance outlier filtering process described in Section~\\ref{sec:preprocessing}. To evaluate the impact of this step on each class, the number of pruned outliers is quantified in Table~\\ref{tab:outlier_quantification}.

                    \\begin{table}[H]
                    \\centering
                    \\caption{Outlier Pruning Statistics per Class}
                    \\label{tab:outlier_quantification}
                    \\begin{tabular}{@{}lcccc@{}}
                    \\toprule
                    \\textbf{Class Name} & \\textbf{Raw Count} & \\textbf{Filtered Count} & \\textbf{Outliers Removed} & \\textbf{Percentage Removed} \\\\ \\midrule
                    Hardhat             & 77,630             & 77,097                  & 533                       & 0.69\\%                      \\\\
                    NO-Hardhat          & 25,230             & 24,965                  & 265                       & 1.05\\%                      \\\\
                    NO-Gloves           & 21,861             & 21,646                  & 215                       & 0.98\\%                      \\\\
                    Safety Vest         & 17,918             & 17,820                  & 98                        & 0.55\\%                      \\\\
                    Gloves              & 12,507             & 12,393                  & 114                       & 0.91\\%                      \\\\
                    NO-Safety Vest      & 4,440              & 4,417                   & 23                        & 0.52\\%                      \\\\ \\midrule
                    \\textbf{Total}      & \\textbf{159,586}   & \\textbf{158,338}        & \\textbf{1,248}            & \\textbf{0.78\\%}             \\\\ \\bottomrule
                    \\end{tabular}
                    \\end{table}

                    Importantly, the outlier removal is relatively uniform across classes, ranging between 0.5\\% and 1.1\\% of the raw sample size. This suggests the filtering does not disproportionately target minority classes, preserving the valuable minority class data.

                    The dataset displays a severe class imbalance. As illustrated in Figure~\\ref{fig:filtered_dist}, the presence of hardhats dominates the dataset (77,097 crops), whereas classes like \\texttt{NO-Safety Vest} contain only 4,417 crops—representing a 17.5:1 ratio. Under such conditions, standard training metrics like raw accuracy can be highly misleading, which we address through detailed performance auditing.

                    \\begin{figure}[H]
                    \\centering
                    \\includegraphics[width=0.6\\textwidth]{images/class_distribution.png}
                    \\caption{Class distribution of the outlier-filtered dataset used for training.}
                    \\label{fig:filtered_dist}
                    \\end{figure}"""

    replacement_table = """                    A total of 159,586 individual object crops are extracted from the dataset folders. 

                    The dataset displays a severe class imbalance. As illustrated in Figure~\\ref{fig:filtered_dist}, the presence of hardhats dominates the dataset (77,630 crops), whereas classes like \\texttt{NO-Safety Vest} contain only 4,440 crops—representing a 17.5:1 ratio. Under such conditions, standard training metrics like raw accuracy can be highly misleading, which we address through detailed performance auditing.

                    \\begin{figure}[H]
                    \\centering
                    \\includegraphics[width=0.6\\textwidth]{images/class_distribution.png}
                    \\caption{Class distribution of the dataset used for training.}
                    \\label{fig:filtered_dist}
                    \\end{figure}"""

    # 4. Training Pipeline Steps
    target_pipeline = """                    \\subsection{Training Pipeline}
                    The offline training pipeline prepares and trains the classifiers through the following sequential steps:
                    \\begin{enumerate}
                        \\item \\textbf{Filter Labels:} \\texttt{filter\\_labels.py} scans the YOLO labels, filters out non-PPE annotations, remaps the class IDs, and saves the new label files to \\texttt{ppe\\_filtered/}.
                        \\item \\textbf{Crop Images:} \\texttt{crop\\_images.py} reads the bounding boxes, crops the objects from the source images, resizes them to $224 \\times 224$ pixels, and organizes them into directories by class.
                        \\item \\textbf{Feature Extraction:} \\texttt{encode.py} passes all crops through a pre-trained, headless ResNet18 model, saving the resulting 512-dimensional features to \\texttt{resnet\\_features.csv}.
                        \\item \\textbf{Outlier Removal:} \\texttt{remove\\_outliers.py} cleans the dataset by removing noisy bounding boxes, writing the output to \\texttt{resnet\\_embeddings\\_filtered.csv}.
                        \\item \\textbf{Train Classifiers:} \\texttt{train.py} trains three independent binary classifiers: Helmet (\\texttt{Hardhat} vs. \\texttt{NO-Hardhat}), Vest (\\texttt{Safety Vest} vs. \\texttt{NO-Safety Vest}), and Glove (\\texttt{Gloves} vs. \\texttt{NO-Gloves}).
                    \\end{enumerate}"""

    replacement_pipeline = """                    \\subsection{Training Pipeline}
                    The offline training pipeline prepares and trains the classifiers through the following sequential steps:
                    \\begin{enumerate}
                        \\item \\textbf{Filter Labels:} \\texttt{filter\\_labels.py} scans the YOLO labels, filters out non-PPE annotations, remaps the class IDs, and saves the new label files to \\texttt{ppe\\_filtered/}.
                        \\item \\textbf{Crop Images:} \\texttt{crop\\_images.py} reads the bounding boxes, crops the objects from the source images, resizes them to $224 \\times 224$ pixels, and organizes them into directories by class.
                        \\item \\textbf{Feature Extraction:} \\texttt{encode.py} passes all crops through a pre-trained, headless ResNet18 model, saving the resulting 512-dimensional features to \\texttt{resnet\\_features.csv}.
                        \\item \\textbf{Train Classifiers:} \\texttt{train.py} trains three independent binary classifiers: Helmet (\\texttt{Hardhat} vs. \\texttt{NO-Hardhat}), Vest (\\texttt{Safety Vest} vs. \\texttt{NO-Safety Vest}), and Glove (\\texttt{Gloves} vs. \\texttt{NO-Gloves}).
                    \\end{enumerate}"""

    # 5. Section 6: Data Preprocessing
    target_preprocessing = """                    \\section{Data Preprocessing}
                    \\label{sec:preprocessing}
                    Data preprocessing operates at two levels: label sanitization and image normalization. After remapping labels, crops are resized to $224 \\times 224$ pixels. To prevent outlier samples (e.g., mislabeled background boxes or partial crops) from corrupting the Random Forest decision boundaries, a centroid-distance outlier detection algorithm is implemented.

                    For each class $k$, the centroid vector $c_k \\in \\mathbb{R}^{512}$ is computed by averaging the 512-dimensional ResNet18 feature vectors of all samples in that class:
                    \\begin{equation}
                    c_k = \\frac{1}{N_k} \\sum_{i=1}^{N_k} x_i^{(k)}
                    \\end{equation}
                    where $N_k$ is the number of instances in class $k$, and $x_i^{(k)}$ is the feature vector of the $i$-th instance.

                    For each sample $x_i^{(k)}$, its Euclidean distance to its class centroid is calculated:
                    \\begin{equation}
                    d(x_i^{(k)}, c_k) = \\| x_i^{(k)} - c_k \\|_2
                    \\end{equation}

                    The mean distance $\\mu_k$ and standard deviation $\\sigma_k$ of these distances are calculated for class $k$. Any sample that lies further than three standard deviations from the centroid is classified as an outlier and pruned:
                    \\begin{equation}
                    \\text{Discard } x_i^{(k)} \\quad \\text{if} \\quad d(x_i^{(k)}, c_k) > \\mu_k + 3\\sigma_k
                    \\end{equation}
                    This step eliminates training noise, clearing boundary regions in the embedding space before classification training begins."""

    replacement_preprocessing = """                    \\section{Data Preprocessing}
                    \\label{sec:preprocessing}
                    Data preprocessing operates at two levels: label sanitization and image normalization. After filtering labels and mapping them to target class IDs, the cropped regions are resized to a consistent resolution of $224 \\times 224$ pixels. This size is required to fit the input dimensions expected by the ResNet18 feature encoder. Normalization is performed using standard ImageNet mean and standard deviation parameters before encoding."""

    # 6. UMAP Section
    target_umap = """                    Applying balanced class weights \\textit{decreased} the recall of the minority class from 86.18\\% to 83.69\\%, and the F1-score from 91.52\\% to 90.01\\%. This occurs because growing unpruned trees (\\texttt{max\\_depth=None}) on artificially weighted samples can cause the splits to overfit to minority class outliers. This result highlights that class weighting is not a universal solution for class imbalance, and its performance must be evaluated empirically.

                    \\begin{figure}[H]
                    \\centering
                    \\includegraphics[width=0.9\\textwidth]{images/umap_comparison.png}
                    \\caption{UMAP projection comparison showing raw embeddings (left) versus outlier-filtered embeddings (right). Outlier pruning cleans cluster boundaries.}
                    \\label{fig:umap}
                    \\end{figure}

                    The impact of the outlier filtering is shown in Figure~\\ref{fig:umap}. The left panel represents the raw UMAP space, where noisy crops blur class separation. The right panel demonstrates that centroid-distance outlier pruning clears these transitional boundary regions, allowing the Random Forest model to establish clean binary classification boundaries."""

    replacement_umap = """                    Applying balanced class weights \\textit{decreased} the recall of the minority class from 86.18\\% to 83.69\\%, and the F1-score from 91.52\\% to 90.01\\%. This occurs because growing unpruned trees (\\texttt{max\\_depth=None}) on artificially weighted samples can cause the splits to overfit to minority class noise. This result highlights that class weighting is not a universal solution for class imbalance, and its performance must be evaluated empirically."""

    # 7. Conclusion Bullet
    target_conclusion = (
        "\\item \\textbf{Class Imbalance Dynamics:} Applying balanced class weights "
        "(\\texttt{class\\_weight='balanced'}) on unpruned trees actually decreased minority "
        "class recall (86.18\\% to 83.69\\%) due to overfitting on minority outliers, "
        "demonstrating that class weighting is not a universal solution for imbalance."
    )
    
    replacement_conclusion = (
        "\\item \\textbf{Class Imbalance Dynamics:} Applying balanced class weights "
        "(\\texttt{class\\_weight='balanced'}) on unpruned trees actually decreased minority "
        "class recall (86.18\\% to 83.69\\%) due to overfitting on minority class noise, "
        "demonstrating that class weighting is not a universal solution for imbalance."
    )

    print("Running replacements...")
    
    replacements = [
        (target_abstract, replacement_abstract),
        (target_objectives, replacement_objectives),
        (target_table, replacement_table),
        (target_pipeline, replacement_pipeline),
        (target_preprocessing, replacement_preprocessing),
        (target_umap, replacement_umap),
        (target_conclusion, replacement_conclusion),
    ]

    for i, (tgt, rep) in enumerate(replacements):
        # Clean target and replacement lines slightly to be robust
        tgt_clean = tgt.strip()
        rep_clean = rep.strip()
        
        # If the clean target is not in the content, try finding it without lead spaces line-by-line
        if tgt_clean in content:
            content = content.replace(tgt_clean, rep_clean)
            print(f"Replacement {i+1} succeeded using clean strip.")
        elif tgt in content:
            content = content.replace(tgt, rep)
            print(f"Replacement {i+1} succeeded using exact target.")
        else:
            # Fallback: remove leading whitespace from target block lines and search
            print(f"Replacement {i+1} failed to match directly. Searching robustly...")
            # Let's locate it by splitting lines
            lines_tgt = [l.strip() for l in tgt.split("\n") if l.strip()]
            
            # Find sequence of lines matching lines_tgt in content
            content_lines = content.split("\n")
            found = False
            for start_idx in range(len(content_lines) - len(lines_tgt) + 1):
                window = [content_lines[start_idx + idx].strip() for idx in range(len(lines_tgt))]
                if window == lines_tgt:
                    # Found the block! Let's replace the lines
                    lead_spaces = len(content_lines[start_idx]) - len(content_lines[start_idx].lstrip())
                    indent = " " * lead_spaces
                    rep_lines = [indent + l.lstrip() for l in rep.split("\n")]
                    content_lines[start_idx:start_idx + len(lines_tgt)] = rep_lines
                    content = "\n".join(content_lines)
                    print(f"Replacement {i+1} succeeded using robust line-matching.")
                    found = True
                    break
            if not found:
                print(f"Replacement {i+1} COULD NOT BE FOUND.")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Done editing report.tex.")

if __name__ == "__main__":
    main()
