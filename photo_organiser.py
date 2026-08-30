import os
import json
import shutil
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import faiss

from insightface.app import FaceAnalysis


# ============================================================
# CONFIG
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)

DATABASE_FOLDER = ".face_database"

NORMAL_INDEX_FILE = "normal_faces.faiss"
OCCLUSION_INDEX_FILE = "occlusion_faces.faiss"
METADATA_FILE = "metadata.json"
INFO_FILE = "database_info.json"

# Normal matching threshold
DEFAULT_THRESHOLD = 0.45

# Occlusion matching threshold
DEFAULT_OCCLUSION_THRESHOLD = 0.35


# ============================================================
# APPLICATION
# ============================================================

class FacePhotoOrganizer:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Face-Based Photo Organizer - FAISS"
        )

        self.root.geometry(
            "1100x850"
        )

        self.root.minsize(
            950,
            800
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.photo_folder = ""

        self.reference_photo_1 = ""
        self.reference_photo_2 = ""

        self.output_folder = ""

        self.face_app = None

        self.normal_index = None
        self.occlusion_index = None

        self.metadata = []

        self.is_processing = False

        self.search_mode = tk.StringVar(
            value="single"
        )

        self.threshold_var = tk.DoubleVar(
            value=DEFAULT_THRESHOLD
        )

        self.occlusion_threshold_var = tk.DoubleVar(
            value=DEFAULT_OCCLUSION_THRESHOLD
        )

        self.use_occlusion = tk.BooleanVar(
            value=True
        )

        self.create_ui()


    # ========================================================
    # USER INTERFACE
    # ========================================================

    def create_ui(self):

        # ====================================================
        # TITLE
        # ====================================================

        tk.Label(
            self.root,
            text="Face-Based Photo Organizer",
            font=("Arial", 21, "bold")
        ).pack(
            pady=(10, 2)
        )

        tk.Label(
            self.root,
            text="Fast FAISS Face Search",
            font=("Arial", 10)
        ).pack(
            pady=(0, 7)
        )


        # ====================================================
        # PHOTO FOLDER
        # ====================================================

        folder_frame = tk.Frame(
            self.root
        )

        folder_frame.pack(
            fill="x",
            padx=25,
            pady=3
        )

        tk.Label(
            folder_frame,
            text="Photo Folder:",
            width=23,
            anchor="w",
            font=("Arial", 10, "bold")
        ).pack(
            side="left"
        )

        self.folder_label = tk.Label(
            folder_frame,
            text="No folder selected",
            relief="sunken",
            anchor="w",
            padx=5
        )

        self.folder_label.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5
        )

        self.folder_button = tk.Button(
            folder_frame,
            text="Browse Folder",
            width=17,
            command=self.select_photo_folder
        )

        self.folder_button.pack(
            side="right"
        )


        # ====================================================
        # DATABASE
        # ====================================================

        database_frame = tk.Frame(
            self.root
        )

        database_frame.pack(
            fill="x",
            padx=25,
            pady=3
        )

        tk.Label(
            database_frame,
            text="FAISS Database:",
            width=23,
            anchor="w",
            font=("Arial", 10, "bold")
        ).pack(
            side="left"
        )

        self.database_status = tk.Label(
            database_frame,
            text="No database loaded",
            anchor="w"
        )

        self.database_status.pack(
            side="left",
            fill="x",
            expand=True
        )

        self.build_button = tk.Button(
            database_frame,
            text="Build Database",
            width=19,
            command=self.start_database_build
        )

        self.build_button.pack(
            side="right"
        )


        # ====================================================
        # SEARCH MODE
        # ====================================================

        mode_frame = tk.LabelFrame(
            self.root,
            text=" Search Mode ",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )

        mode_frame.pack(
            fill="x",
            padx=25,
            pady=5
        )

        tk.Radiobutton(
            mode_frame,
            text="1 Person",
            variable=self.search_mode,
            value="single",
            command=self.update_reference_ui,
            font=("Arial", 9)
        ).pack(
            side="left",
            padx=15
        )

        tk.Radiobutton(
            mode_frame,
            text="2 Persons - Couple Photo",
            variable=self.search_mode,
            value="couple",
            command=self.update_reference_ui,
            font=("Arial", 9)
        ).pack(
            side="left",
            padx=15
        )

        tk.Radiobutton(
            mode_frame,
            text="2 Persons - Different Photos",
            variable=self.search_mode,
            value="individual",
            command=self.update_reference_ui,
            font=("Arial", 9)
        ).pack(
            side="left",
            padx=15
        )


        # ====================================================
        # REFERENCE PHOTO AREA
        # ====================================================

        self.reference_container = tk.Frame(
            self.root
        )

        self.reference_container.pack(
            fill="x",
            padx=25,
            pady=2
        )


        # ----------------------------------------------------
        # PERSON 1
        # ----------------------------------------------------

        self.reference_row_1 = tk.Frame(
            self.reference_container
        )

        self.reference_row_1.pack(
            fill="x",
            pady=2
        )

        self.reference_title_1 = tk.Label(
            self.reference_row_1,
            text="Person Photo:",
            width=23,
            anchor="w",
            font=("Arial", 10, "bold")
        )

        self.reference_title_1.pack(
            side="left"
        )

        self.reference_label_1 = tk.Label(
            self.reference_row_1,
            text="No photo selected",
            relief="sunken",
            anchor="w",
            padx=5
        )

        self.reference_label_1.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5
        )

        self.reference_button_1 = tk.Button(
            self.reference_row_1,
            text="Browse Photo",
            width=17,
            command=self.select_reference_1
        )

        self.reference_button_1.pack(
            side="right"
        )


        # ----------------------------------------------------
        # PERSON 2
        # ----------------------------------------------------

        self.reference_row_2 = tk.Frame(
            self.reference_container
        )

        self.reference_title_2 = tk.Label(
            self.reference_row_2,
            text="Person 2 Photo:",
            width=23,
            anchor="w",
            font=("Arial", 10, "bold")
        )

        self.reference_title_2.pack(
            side="left"
        )

        self.reference_label_2 = tk.Label(
            self.reference_row_2,
            text="No photo selected",
            relief="sunken",
            anchor="w",
            padx=5
        )

        self.reference_label_2.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5
        )

        self.reference_button_2 = tk.Button(
            self.reference_row_2,
            text="Browse Photo",
            width=17,
            command=self.select_reference_2
        )

        self.reference_button_2.pack(
            side="right"
        )


        # ====================================================
        # NOTE
        # ====================================================

        tk.Label(
            self.root,
            text=(
                "NOTE: Photos with sunglasses or goggles "
                "may not match well"
            ),
            font=("Arial", 9, "italic")
        ).pack(
            pady=(3, 4)
        )


        # ====================================================
        # OCCLUSION OPTION
        # ====================================================

        occlusion_frame = tk.Frame(
            self.root
        )

        occlusion_frame.pack(
            fill="x",
            padx=25,
            pady=2
        )

        tk.Checkbutton(
            occlusion_frame,
            text=(
                "Use sunglasses / goggles tolerant search"
            ),
            variable=self.use_occlusion,
            font=("Arial", 9)
        ).pack(
            anchor="w"
        )


        # ====================================================
        # OUTPUT
        # ====================================================

        output_frame = tk.Frame(
            self.root
        )

        output_frame.pack(
            fill="x",
            padx=25,
            pady=3
        )

        tk.Label(
            output_frame,
            text="Output Folder:",
            width=23,
            anchor="w",
            font=("Arial", 10, "bold")
        ).pack(
            side="left"
        )

        self.output_entry = tk.Entry(
            output_frame
        )

        self.output_entry.insert(
            0,
            "Matched Photos"
        )

        self.output_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5
        )


        # ====================================================
        # THRESHOLD
        # ====================================================

        threshold_frame = tk.Frame(
            self.root
        )

        threshold_frame.pack(
            fill="x",
            padx=25,
            pady=2
        )

        tk.Label(
            threshold_frame,
            text="Normal Match Threshold:",
            width=23,
            anchor="w",
            font=("Arial", 10, "bold")
        ).pack(
            side="left"
        )

        self.threshold_scale = tk.Scale(
            threshold_frame,
            from_=0.25,
            to=0.70,
            resolution=0.01,
            orient="horizontal",
            length=280,
            variable=self.threshold_var,
            command=self.update_threshold,
            showvalue=False
        )

        self.threshold_scale.pack(
            side="left"
        )

        self.threshold_label = tk.Label(
            threshold_frame,
            text=f"{DEFAULT_THRESHOLD:.2f}",
            width=5
        )

        self.threshold_label.pack(
            side="left",
            padx=5
        )


        # ====================================================
        # OCCLUSION THRESHOLD
        # ====================================================

        occlusion_threshold_frame = tk.Frame(
            self.root
        )

        occlusion_threshold_frame.pack(
            fill="x",
            padx=25,
            pady=2
        )

        tk.Label(
            occlusion_threshold_frame,
            text="Goggles Match Threshold:",
            width=23,
            anchor="w",
            font=("Arial", 10, "bold")
        ).pack(
            side="left"
        )

        self.occlusion_scale = tk.Scale(
            occlusion_threshold_frame,
            from_=0.20,
            to=0.60,
            resolution=0.01,
            orient="horizontal",
            length=280,
            variable=self.occlusion_threshold_var,
            command=self.update_occlusion_threshold,
            showvalue=False
        )

        self.occlusion_scale.pack(
            side="left"
        )

        self.occlusion_threshold_label = tk.Label(
            occlusion_threshold_frame,
            text=f"{DEFAULT_OCCLUSION_THRESHOLD:.2f}",
            width=5
        )

        self.occlusion_threshold_label.pack(
            side="left",
            padx=5
        )


        # ====================================================
        # FIND BUTTON
        # ====================================================

        self.search_button = tk.Button(
            self.root,
            text="FIND PHOTOS",
            font=("Arial", 12, "bold"),
            width=22,
            height=1,
            command=self.start_search
        )

        self.search_button.pack(
            pady=6
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            mode="determinate"
        )

        self.progress.pack(
            fill="x",
            padx=25,
            pady=2
        )


        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 9)
        )

        self.status_label.pack(
            pady=2
        )


        # ====================================================
        # PROCESSING LOG
        # ====================================================

        self.log_frame = tk.Frame(
            self.root
        )

        self.log_frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=(3, 10)
        )

        tk.Label(
            self.log_frame,
            text="Processing Log",
            font=("Arial", 10, "bold")
        ).pack(
            anchor="w"
        )

        log_text_frame = tk.Frame(
            self.log_frame
        )

        log_text_frame.pack(
            fill="both",
            expand=True,
            pady=2
        )

        self.log_box = tk.Text(
            log_text_frame,
            height=8,
            state="disabled",
            wrap="word",
            font=("Consolas", 9)
        )

        self.log_box.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar = tk.Scrollbar(
            log_text_frame,
            command=self.log_box.yview
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.log_box.config(
            yscrollcommand=scrollbar.set
        )


        # ====================================================
        # INITIAL STATE
        # ====================================================

        self.update_reference_ui()


    # ========================================================
    # REFERENCE UI
    # ========================================================

    def update_reference_ui(self):

        mode = self.search_mode.get()

        # ----------------------------------------------------
        # ONE PERSON
        # ----------------------------------------------------

        if mode == "single":

            self.reference_title_1.config(
                text="Person Photo:"
            )

            self.reference_row_2.pack_forget()


        # ----------------------------------------------------
        # COUPLE PHOTO
        # ----------------------------------------------------

        elif mode == "couple":

            self.reference_title_1.config(
                text="Couple Photo:"
            )

            self.reference_row_2.pack_forget()


        # ----------------------------------------------------
        # TWO DIFFERENT PHOTOS
        # ----------------------------------------------------

        elif mode == "individual":

            self.reference_title_1.config(
                text="Person 1 Photo:"
            )

            self.reference_title_2.config(
                text="Person 2 Photo:"
            )

            self.reference_row_2.pack(
                fill="x",
                pady=2
            )


    # ========================================================
    # SELECT FOLDER
    # ========================================================

    def select_photo_folder(self):

        folder = filedialog.askdirectory(
            title="Select Photo Folder"
        )

        if not folder:
            return

        self.photo_folder = os.path.abspath(
            folder
        )

        self.folder_label.config(
            text=self.photo_folder
        )

        self.log(
            f"Photo folder selected:\n"
            f"{self.photo_folder}"
        )

        self.check_existing_database()


    # ========================================================
    # REFERENCE 1
    # ========================================================

    def select_reference_1(self):

        file = filedialog.askopenfilename(
            title="Select Reference Photo",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"
                )
            ]
        )

        if not file:
            return

        self.reference_photo_1 = os.path.abspath(
            file
        )

        self.reference_label_1.config(
            text=self.reference_photo_1
        )

        self.log(
            f"Reference 1 selected:\n"
            f"{self.reference_photo_1}"
        )


    # ========================================================
    # REFERENCE 2
    # ========================================================

    def select_reference_2(self):

        file = filedialog.askopenfilename(
            title="Select Person 2 Photo",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"
                )
            ]
        )

        if not file:
            return

        self.reference_photo_2 = os.path.abspath(
            file
        )

        self.reference_label_2.config(
            text=self.reference_photo_2
        )

        self.log(
            f"Reference 2 selected:\n"
            f"{self.reference_photo_2}"
        )


    # ========================================================
    # DATABASE PATHS
    # ========================================================

    def get_database_folder(self):

        return os.path.join(
            self.photo_folder,
            DATABASE_FOLDER
        )


    def get_normal_index_path(self):

        return os.path.join(
            self.get_database_folder(),
            NORMAL_INDEX_FILE
        )


    def get_occlusion_index_path(self):

        return os.path.join(
            self.get_database_folder(),
            OCCLUSION_INDEX_FILE
        )


    def get_metadata_path(self):

        return os.path.join(
            self.get_database_folder(),
            METADATA_FILE
        )


    def get_info_path(self):

        return os.path.join(
            self.get_database_folder(),
            INFO_FILE
        )


    # ========================================================
    # LOAD DATABASE
    # ========================================================

    def load_database(self):

        normal_path = (
            self.get_normal_index_path()
        )

        occlusion_path = (
            self.get_occlusion_index_path()
        )

        metadata_path = (
            self.get_metadata_path()
        )

        if not os.path.exists(
            normal_path
        ):

            raise Exception(
                f"Normal FAISS index not found:\n"
                f"{normal_path}"
            )

        if not os.path.exists(
            occlusion_path
        ):

            raise Exception(
                f"Occlusion FAISS index not found:\n"
                f"{occlusion_path}"
            )

        if not os.path.exists(
            metadata_path
        ):

            raise Exception(
                f"Metadata file not found:\n"
                f"{metadata_path}"
            )

        self.normal_index = (
            faiss.read_index(
                normal_path
            )
        )

        self.occlusion_index = (
            faiss.read_index(
                occlusion_path
            )
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(
                file
            )

        if (
            self.normal_index.ntotal
            != len(self.metadata)
        ):

            raise Exception(
                "Normal FAISS index and metadata "
                "do not match."
            )

        if (
            self.occlusion_index.ntotal
            != len(self.metadata)
        ):

            raise Exception(
                "Occlusion FAISS index and metadata "
                "do not match."
            )

        return True


    # ========================================================
    # CHECK DATABASE
    # ========================================================

    def check_existing_database(self):

        try:

            self.load_database()

            photos = len(
                set(
                    item["photo"]
                    for item in self.metadata
                )
            )

            vectors = (
                self.normal_index.ntotal
            )

            self.database_status.config(
                text=(
                    f"✓ Ready | "
                    f"{photos} photos | "
                    f"{vectors} face vectors"
                )
            )

            self.log(
                "Existing FAISS database loaded."
            )

        except Exception as error:

            self.normal_index = None
            self.occlusion_index = None
            self.metadata = []

            self.database_status.config(
                text="No database loaded"
            )

            self.log(
                f"No usable database:\n{error}"
            )


    # ========================================================
    # LOAD INSIGHTFACE
    # ========================================================

    def load_model(self):

        if self.face_app is not None:
            return

        self.set_status(
            "Loading InsightFace model..."
        )

        self.log(
            "Loading InsightFace..."
        )

        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        self.face_app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        self.log(
            "InsightFace loaded."
        )


    # ========================================================
    # NORMALIZE EMBEDDING
    # ========================================================

    @staticmethod
    def normalize_embedding(
        embedding
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:

            raise Exception(
                "Invalid embedding."
            )

        return embedding / norm


    # ========================================================
    # READ IMAGE
    # ========================================================

    def read_image(
        self,
        path
    ):

        image = cv2.imread(
            path
        )

        if image is None:

            raise Exception(
                f"Could not read image:\n{path}"
            )

        return image


    # ========================================================
    # GET FACES
    # ========================================================

    def get_faces(
        self,
        image
    ):

        return self.face_app.get(
            image
        )


    # ========================================================
    # GET LARGEST FACE
    # ========================================================

    @staticmethod
    def largest_face(
        faces
    ):

        if not faces:
            return None

        return max(
            faces,
            key=lambda f:
            (
                f.bbox[2] - f.bbox[0]
            )
            *
            (
                f.bbox[3] - f.bbox[1]
            )
        )


    # ========================================================
    # CREATE EYE OCCLUDED IMAGE
    # ========================================================

    def create_eye_occluded_image(
        self,
        image,
        face
    ):

        result = image.copy()

        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        h, w = image.shape[:2]

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            w - 1,
            x2
        )

        y2 = min(
            h - 1,
            y2
        )

        face_width = (
            x2 - x1
        )

        face_height = (
            y2 - y1
        )

        if (
            face_width <= 0
            or
            face_height <= 0
        ):

            return result

        # Eye / sunglasses region
        left = int(
            x1 +
            face_width * 0.08
        )

        right = int(
            x1 +
            face_width * 0.92
        )

        top = int(
            y1 +
            face_height * 0.20
        )

        bottom = int(
            y1 +
            face_height * 0.52
        )

        left = max(
            x1,
            left
        )

        right = min(
            x2,
            right
        )

        top = max(
            y1,
            top
        )

        bottom = min(
            y2,
            bottom
        )

        region = result[
            top:bottom,
            left:right
        ]

        if region.size > 0:

            blurred = cv2.GaussianBlur(
                region,
                (31, 31),
                0
            )

            result[
                top:bottom,
                left:right
            ] = blurred

        return result


    # ========================================================
    # FACE EMBEDDINGS FROM IMAGE
    # ========================================================

    def get_face_embeddings_from_image(
        self,
        image
    ):

        faces = self.get_faces(
            image
        )

        results = []

        for face in faces:

            normal = (
                self.normalize_embedding(
                    face.embedding
                )
            )

            occluded_image = (
                self.create_eye_occluded_image(
                    image,
                    face
                )
            )

            occluded_faces = (
                self.get_faces(
                    occluded_image
                )
            )

            occluded_face = (
                self.largest_face(
                    occluded_faces
                )
            )

            if occluded_face is not None:

                occlusion = (
                    self.normalize_embedding(
                        occluded_face.embedding
                    )
                )

            else:

                # Fallback to normal embedding
                occlusion = normal.copy()

            results.append(
                {
                    "normal": normal,
                    "occlusion": occlusion
                }
            )

        return results


    # ========================================================
    # REFERENCE SINGLE EMBEDDING
    # ========================================================

    def get_reference_embeddings(
        self,
        path
    ):

        image = self.read_image(
            path
        )

        embeddings = (
            self.get_face_embeddings_from_image(
                image
            )
        )

        if len(embeddings) == 0:

            raise Exception(
                "No face detected in reference photo."
            )

        if len(embeddings) > 1:

            self.log(
                f"Warning: {len(embeddings)} faces detected."
            )

            self.log(
                "Using largest face."
            )

            faces = self.get_faces(
                image
            )

            largest = self.largest_face(
                faces
            )

            largest_index = 0

            largest_area = -1

            for i, face in enumerate(
                faces
            ):

                x1, y1, x2, y2 = (
                    face.bbox
                )

                area = (
                    x2 - x1
                ) * (
                    y2 - y1
                )

                if area > largest_area:

                    largest_area = area
                    largest_index = i

            return embeddings[
                largest_index
            ]

        return embeddings[0]


    # ========================================================
    # COUPLE EMBEDDINGS
    # ========================================================

    def get_couple_reference_embeddings(
        self,
        path
    ):

        image = self.read_image(
            path
        )

        faces = self.get_faces(
            image
        )

        if len(faces) != 2:

            raise Exception(
                "Couple photo must contain exactly "
                "2 faces.\n\n"
                f"Detected: {len(faces)}"
            )

        # Sort faces from left to right
        faces = sorted(
            faces,
            key=lambda f:
            f.bbox[0]
        )

        results = []

        for face in faces:

            normal = (
                self.normalize_embedding(
                    face.embedding
                )
            )

            occluded_image = (
                self.create_eye_occluded_image(
                    image,
                    face
                )
            )

            occluded_faces = (
                self.get_faces(
                    occluded_image
                )
            )

            if occluded_faces:

                closest = min(
                    occluded_faces,
                    key=lambda f:
                    abs(
                        (
                            f.bbox[0]
                            +
                            f.bbox[2]
                        ) / 2
                        -
                        (
                            face.bbox[0]
                            +
                            face.bbox[2]
                        ) / 2
                    )
                )

                occlusion = (
                    self.normalize_embedding(
                        closest.embedding
                    )
                )

            else:

                occlusion = normal.copy()

            results.append(
                {
                    "normal": normal,
                    "occlusion": occlusion
                }
            )

        return results


    # ========================================================
    # FAISS SEARCH
    # ========================================================

    def search_index(
        self,
        index,
        embedding,
        threshold
    ):

        if index is None:
            return {}

        total = index.ntotal

        if total == 0:
            return {}

        query = np.asarray(
            [embedding],
            dtype=np.float32
        )

        faiss.normalize_L2(
            query
        )

        scores, indices = (
            index.search(
                query,
                total
            )
        )

        results = {}

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            similarity = float(
                score
            )

            if similarity < threshold:
                break

            photo = (
                self.metadata[
                    int(idx)
                ]["photo"]
            )

            if (
                photo not in results
                or
                similarity >
                results[photo]
            ):

                results[photo] = similarity

        return results


    # ========================================================
    # COMBINE NORMAL + OCCLUSION RESULTS
    # ========================================================

    def combined_search(
        self,
        reference,
        normal_threshold,
        occlusion_threshold
    ):

        # ----------------------------------------------------
        # NORMAL FAISS
        # ----------------------------------------------------

        normal_results = (
            self.search_index(
                self.normal_index,
                reference["normal"],
                normal_threshold
            )
        )

        # ----------------------------------------------------
        # OCCLUSION FAISS
        # ----------------------------------------------------

        if self.use_occlusion.get():

            occlusion_results = (
                self.search_index(
                    self.occlusion_index,
                    reference["occlusion"],
                    occlusion_threshold
                )
            )

        else:

            occlusion_results = {}


        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------

        combined = {}

        for photo, score in (
            normal_results.items()
        ):

            combined[photo] = score


        for photo, score in (
            occlusion_results.items()
        ):

            # Keep highest confidence
            if (
                photo not in combined
                or
                score > combined[photo]
            ):

                combined[photo] = score


        self.log(
            f"Normal FAISS matches: "
            f"{len(normal_results)}"
        )

        if self.use_occlusion.get():

            self.log(
                f"Occlusion FAISS matches: "
                f"{len(occlusion_results)}"
            )

        self.log(
            f"Combined matches: "
            f"{len(combined)}"
        )

        return combined


    # ========================================================
    # BOTH PEOPLE
    # ========================================================

    def find_common_photos(
        self,
        results_1,
        results_2
    ):

        common = (
            set(results_1.keys())
            &
            set(results_2.keys())
        )

        result = {}

        for photo in common:

            result[photo] = {
                "score_1": results_1[photo],
                "score_2": results_2[photo]
            }

        return result


    # ========================================================
    # START SEARCH
    # ========================================================

    def start_search(self):

        if self.is_processing:
            return

        if not self.photo_folder:

            messagebox.showwarning(
                "Photo Folder",
                "Please select the photo folder."
            )

            return

        if self.normal_index is None:

            try:

                self.load_database()

            except Exception as error:

                messagebox.showerror(
                    "Database Error",
                    str(error)
                )

                return

        mode = self.search_mode.get()

        # ----------------------------------------------------
        # SINGLE
        # ----------------------------------------------------

        if mode == "single":

            if not self.reference_photo_1:

                messagebox.showwarning(
                    "Reference Photo",
                    "Please select a person photo."
                )

                return


        # ----------------------------------------------------
        # COUPLE
        # ----------------------------------------------------

        elif mode == "couple":

            if not self.reference_photo_1:

                messagebox.showwarning(
                    "Couple Photo",
                    "Please select the couple photo."
                )

                return


        # ----------------------------------------------------
        # TWO INDIVIDUAL
        # ----------------------------------------------------

        elif mode == "individual":

            if not self.reference_photo_1:

                messagebox.showwarning(
                    "Person 1",
                    "Please select Person 1 photo."
                )

                return

            if not self.reference_photo_2:

                messagebox.showwarning(
                    "Person 2",
                    "Please select Person 2 photo."
                )

                return


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        output_name = (
            self.output_entry.get().strip()
        )

        if not output_name:

            messagebox.showwarning(
                "Output",
                "Please enter output folder name."
            )

            return

        self.output_folder = os.path.join(
            self.photo_folder,
            output_name
        )

        self.is_processing = True

        self.disable_buttons()

        threading.Thread(
            target=self.perform_search,
            daemon=True
        ).start()


    # ========================================================
    # SEARCH
    # ========================================================

    def perform_search(self):

        try:

            normal_threshold = (
                self.threshold_var.get()
            )

            occlusion_threshold = (
                self.occlusion_threshold_var.get()
            )

            mode = self.search_mode.get()

            self.load_model()

            self.log("")
            self.log(
                "========================================"
            )

            self.log(
                "SEARCH STARTED"
            )

            self.log(
                f"Mode: {mode}"
            )

            self.log(
                f"Normal threshold: "
                f"{normal_threshold:.2f}"
            )

            self.log(
                f"Goggles threshold: "
                f"{occlusion_threshold:.2f}"
            )

            self.log(
                "Using FAISS only - "
                "no full folder rescan."
            )

            self.log(
                "========================================"
            )


            # =================================================
            # SINGLE PERSON
            # =================================================

            if mode == "single":

                self.set_status(
                    "Analyzing reference photo..."
                )

                reference = (
                    self.get_reference_embeddings(
                        self.reference_photo_1
                    )
                )

                self.set_status(
                    "Searching FAISS..."
                )

                matches = (
                    self.combined_search(
                        reference,
                        normal_threshold,
                        occlusion_threshold
                    )
                )

                final_results = {}

                for photo, score in (
                    matches.items()
                ):

                    final_results[photo] = {
                        "score": score
                    }


            # =================================================
            # COUPLE PHOTO
            # =================================================

            elif mode == "couple":

                self.set_status(
                    "Analyzing couple photo..."
                )

                references = (
                    self.get_couple_reference_embeddings(
                        self.reference_photo_1
                    )
                )

                self.log(
                    "Exactly 2 faces detected."
                )

                # ------------------------------------------------
                # PERSON 1
                # ------------------------------------------------

                self.set_status(
                    "Searching Person 1..."
                )

                results_1 = (
                    self.combined_search(
                        references[0],
                        normal_threshold,
                        occlusion_threshold
                    )
                )

                # ------------------------------------------------
                # PERSON 2
                # ------------------------------------------------

                self.set_status(
                    "Searching Person 2..."
                )

                results_2 = (
                    self.combined_search(
                        references[1],
                        normal_threshold,
                        occlusion_threshold
                    )
                )

                # ------------------------------------------------
                # BOTH
                # ------------------------------------------------

                self.set_status(
                    "Finding photos containing both..."
                )

                final_results = (
                    self.find_common_photos(
                        results_1,
                        results_2
                    )
                )

                self.log(
                    f"Photos containing both: "
                    f"{len(final_results)}"
                )


            # =================================================
            # TWO DIFFERENT PHOTOS
            # =================================================

            elif mode == "individual":

                self.set_status(
                    "Analyzing Person 1..."
                )

                reference_1 = (
                    self.get_reference_embeddings(
                        self.reference_photo_1
                    )
                )

                self.set_status(
                    "Analyzing Person 2..."
                )

                reference_2 = (
                    self.get_reference_embeddings(
                        self.reference_photo_2
                    )
                )

                # ------------------------------------------------
                # PERSON 1
                # ------------------------------------------------

                self.set_status(
                    "Searching Person 1..."
                )

                results_1 = (
                    self.combined_search(
                        reference_1,
                        normal_threshold,
                        occlusion_threshold
                    )
                )

                # ------------------------------------------------
                # PERSON 2
                # ------------------------------------------------

                self.set_status(
                    "Searching Person 2..."
                )

                results_2 = (
                    self.combined_search(
                        reference_2,
                        normal_threshold,
                        occlusion_threshold
                    )
                )

                # ------------------------------------------------
                # BOTH
                # ------------------------------------------------

                self.set_status(
                    "Finding photos containing both..."
                )

                final_results = (
                    self.find_common_photos(
                        results_1,
                        results_2
                    )
                )

                self.log(
                    f"Photos containing both: "
                    f"{len(final_results)}"
                )


            else:

                raise Exception(
                    "Invalid search mode."
                )


            # =================================================
            # SORT
            # =================================================

            if mode == "single":

                sorted_results = sorted(
                    final_results.items(),
                    key=lambda x:
                    x[1]["score"],
                    reverse=True
                )

            else:

                sorted_results = sorted(
                    final_results.items(),
                    key=lambda x:
                    (
                        x[1]["score_1"]
                        +
                        x[1]["score_2"]
                    ) / 2,
                    reverse=True
                )


            # =================================================
            # COPY RESULTS
            # =================================================

            os.makedirs(
                self.output_folder,
                exist_ok=True
            )

            total = len(
                sorted_results
            )

            self.root.after(
                0,
                lambda: self.progress.config(
                    maximum=max(total, 1),
                    value=0
                )
            )

            copied = 0

            for number, (
                photo,
                scores
            ) in enumerate(
                sorted_results,
                start=1
            ):

                if not os.path.isfile(
                    photo
                ):

                    self.log(
                        f"[MISSING] {photo}"
                    )

                    continue

                try:

                    self.copy_photo(
                        photo,
                        self.output_folder
                    )

                    copied += 1

                    filename = (
                        os.path.basename(
                            photo
                        )
                    )

                    if mode == "single":

                        self.log(
                            f"[MATCH "
                            f"{scores['score']:.3f}] "
                            f"{filename}"
                        )

                    else:

                        self.log(
                            f"[BOTH "
                            f"{scores['score_1']:.3f} / "
                            f"{scores['score_2']:.3f}] "
                            f"{filename}"
                        )

                except Exception as error:

                    self.log(
                        f"[COPY ERROR] "
                        f"{photo}: {error}"
                    )

                self.root.after(
                    0,
                    lambda value=number:
                    self.progress.config(
                        value=value
                    )
                )


            # =================================================
            # DONE
            # =================================================

            self.set_status(
                f"Done - {copied} photos copied"
            )

            self.log("")
            self.log(
                "========================================"
            )

            self.log(
                "SEARCH COMPLETE"
            )

            self.log(
                f"Matching photos: "
                f"{len(final_results)}"
            )

            self.log(
                f"Photos copied: "
                f"{copied}"
            )

            self.log(
                f"Output: "
                f"{self.output_folder}"
            )

            self.log(
                "========================================"
            )

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Search Complete",
                    f"Search completed!\n\n"
                    f"Matching photos: "
                    f"{len(final_results)}\n"
                    f"Photos copied: {copied}\n\n"
                    f"Output:\n"
                    f"{self.output_folder}"
                )
            )

        except Exception as error:

            self.log(
                traceback.format_exc()
            )

            self.set_status(
                "Search failed."
            )

            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Search Error",
                    str(error)
                )
            )

        finally:

            self.is_processing = False

            self.enable_buttons()


    # ========================================================
    # COPY PHOTO
    # ========================================================

    def copy_photo(
        self,
        source,
        destination_folder
    ):

        source = os.path.abspath(
            os.path.normpath(
                source
            )
        )

        destination_folder = os.path.abspath(
            os.path.normpath(
                destination_folder
            )
        )

        if not os.path.isfile(
            source
        ):

            raise FileNotFoundError(
                f"Photo not found:\n{source}"
            )

        os.makedirs(
            destination_folder,
            exist_ok=True
        )

        filename = os.path.basename(
            source
        )

        destination = os.path.join(
            destination_folder,
            filename
        )

        if os.path.exists(
            destination
        ):

            name, extension = (
                os.path.splitext(
                    filename
                )
            )

            counter = 1

            while os.path.exists(
                destination
            ):

                destination = os.path.join(
                    destination_folder,
                    f"{name}_{counter}{extension}"
                )

                counter += 1

        shutil.copy2(
            source,
            destination
        )


    # ========================================================
    # BUILD DATABASE
    # ========================================================

    def start_database_build(self):

        if self.is_processing:
            return

        if not self.photo_folder:

            messagebox.showwarning(
                "Photo Folder",
                "Please select the photo folder first."
            )

            return

        confirm = messagebox.askyesno(
            "Build FAISS Database",
            "Build a new face database from this folder?\n\n"
            "This may take some time for a large photo collection."
        )

        if not confirm:
            return

        self.is_processing = True

        self.disable_buttons()

        threading.Thread(
            target=self.build_database,
            daemon=True
        ).start()


    # ========================================================
    # BUILD DATABASE
    # ========================================================

    def build_database(self):

        try:

            self.load_model()

            database_folder = (
                self.get_database_folder()
            )

            os.makedirs(
                database_folder,
                exist_ok=True
            )

            # ------------------------------------------------
            # FIND PHOTOS
            # ------------------------------------------------

            image_files = []

            for root, dirs, files in os.walk(
                self.photo_folder
            ):

                # Do not scan database folder
                dirs[:] = [
                    d
                    for d in dirs
                    if os.path.abspath(
                        os.path.join(
                            root,
                            d
                        )
                    )
                    != os.path.abspath(
                        database_folder
                    )
                ]

                for filename in files:

                    if filename.lower().endswith(
                        SUPPORTED_EXTENSIONS
                    ):

                        image_files.append(
                            os.path.abspath(
                                os.path.join(
                                    root,
                                    filename
                                )
                            )
                        )

            image_files.sort()

            self.log("")
            self.log(
                "========================================"
            )

            self.log(
                "DATABASE BUILD STARTED"
            )

            self.log(
                f"Photos found: "
                f"{len(image_files)}"
            )

            self.log(
                "Creating normal + "
                "occlusion embeddings."
            )

            self.log(
                "========================================"
            )


            # ------------------------------------------------
            # ARRAYS
            # ------------------------------------------------

            normal_vectors = []
            occlusion_vectors = []
            metadata = []


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            total = len(
                image_files
            )

            self.root.after(
                0,
                lambda: self.progress.config(
                    maximum=max(total, 1),
                    value=0
                )
            )


            # ------------------------------------------------
            # PROCESS PHOTOS
            # ------------------------------------------------

            successful = 0
            failed = 0

            for number, path in enumerate(
                image_files,
                start=1
            ):

                filename = (
                    os.path.basename(
                        path
                    )
                )

                self.set_status(
                    f"Building database "
                    f"{number}/{total}"
                )

                self.log(
                    f"[{number}/{total}] "
                    f"{filename}"
                )

                try:

                    image = self.read_image(
                        path
                    )

                    face_embeddings = (
                        self.get_face_embeddings_from_image(
                            image
                        )
                    )

                    self.log(
                        f"  Faces detected: "
                        f"{len(face_embeddings)}"
                    )

                    for face_number, data in enumerate(
                        face_embeddings
                    ):

                        normal_vectors.append(
                            data["normal"]
                        )

                        occlusion_vectors.append(
                            data["occlusion"]
                        )

                        metadata.append(
                            {
                                "photo": path,
                                "face_number": face_number
                            }
                        )

                    successful += 1

                except Exception as error:

                    failed += 1

                    self.log(
                        f"  ERROR: {error}"
                    )

                self.root.after(
                    0,
                    lambda value=number:
                    self.progress.config(
                        value=value
                    )
                )


            # ------------------------------------------------
            # VALIDATE
            # ------------------------------------------------

            if not normal_vectors:

                raise Exception(
                    "No faces were detected in the "
                    "selected photo folder."
                )


            # ------------------------------------------------
            # CREATE ARRAYS
            # ------------------------------------------------

            normal_matrix = np.asarray(
                normal_vectors,
                dtype=np.float32
            )

            occlusion_matrix = np.asarray(
                occlusion_vectors,
                dtype=np.float32
            )

            # Normalize
            faiss.normalize_L2(
                normal_matrix
            )

            faiss.normalize_L2(
                occlusion_matrix
            )


            # ------------------------------------------------
            # NORMAL INDEX
            # ------------------------------------------------

            self.set_status(
                "Creating normal FAISS index..."
            )

            normal_index = faiss.IndexFlatIP(
                normal_matrix.shape[1]
            )

            normal_index.add(
                normal_matrix
            )


            # ------------------------------------------------
            # OCCLUSION INDEX
            # ------------------------------------------------

            self.set_status(
                "Creating goggles FAISS index..."
            )

            occlusion_index = faiss.IndexFlatIP(
                occlusion_matrix.shape[1]
            )

            occlusion_index.add(
                occlusion_matrix
            )


            # ------------------------------------------------
            # SAVE INDEXES
            # ------------------------------------------------

            faiss.write_index(
                normal_index,
                self.get_normal_index_path()
            )

            faiss.write_index(
                occlusion_index,
                self.get_occlusion_index_path()
            )


            # ------------------------------------------------
            # SAVE METADATA
            # ------------------------------------------------

            with open(
                self.get_metadata_path(),
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=2
                )


            # ------------------------------------------------
            # SAVE INFO
            # ------------------------------------------------

            unique_photos = len(
                set(
                    item["photo"]
                    for item in metadata
                )
            )

            info = {
                "version": 2,
                "photo_count": unique_photos,
                "face_vector_count": len(
                    metadata
                ),
                "normal_index": NORMAL_INDEX_FILE,
                "occlusion_index": OCCLUSION_INDEX_FILE,
                "metadata": METADATA_FILE
            }

            with open(
                self.get_info_path(),
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    info,
                    file,
                    indent=2
                )


            # ------------------------------------------------
            # ASSIGN TO APP
            # ------------------------------------------------

            self.normal_index = (
                normal_index
            )

            self.occlusion_index = (
                occlusion_index
            )

            self.metadata = metadata


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            self.database_status.config(
                text=(
                    f"✓ Ready | "
                    f"{unique_photos} photos | "
                    f"{len(metadata)} face vectors"
                )
            )

            self.set_status(
                "Database ready."
            )

            self.log("")
            self.log(
                "========================================"
            )

            self.log(
                "DATABASE BUILD COMPLETE"
            )

            self.log(
                f"Photos processed: "
                f"{successful}"
            )

            self.log(
                f"Failed photos: "
                f"{failed}"
            )

            self.log(
                f"Unique photos: "
                f"{unique_photos}"
            )

            self.log(
                f"Face vectors: "
                f"{len(metadata)}"
            )

            self.log(
                f"Normal FAISS vectors: "
                f"{normal_index.ntotal}"
            )

            self.log(
                f"Occlusion FAISS vectors: "
                f"{occlusion_index.ntotal}"
            )

            self.log(
                "========================================"
            )

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Database Ready",
                    f"Database created successfully!\n\n"
                    f"Photos: {unique_photos}\n"
                    f"Face vectors: {len(metadata)}\n\n"
                    f"Normal + goggles indexes created."
                )
            )

        except Exception as error:

            self.log(
                traceback.format_exc()
            )

            self.set_status(
                "Database build failed."
            )

            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Database Error",
                    str(error)
                )
            )

        finally:

            self.is_processing = False

            self.enable_buttons()


    # ========================================================
    # LOG
    # ========================================================

    def log(
        self,
        text
    ):

        def update():

            self.log_box.config(
                state="normal"
            )

            self.log_box.insert(
                "end",
                text + "\n"
            )

            self.log_box.see(
                "end"
            )

            self.log_box.config(
                state="disabled"
            )

        self.root.after(
            0,
            update
        )


    # ========================================================
    # STATUS
    # ========================================================

    def set_status(
        self,
        text
    ):

        self.root.after(
            0,
            lambda: self.status_label.config(
                text=text
            )
        )


    # ========================================================
    # THRESHOLD
    # ========================================================

    def update_threshold(
        self,
        value
    ):

        self.threshold_label.config(
            text=f"{float(value):.2f}"
        )


    def update_occlusion_threshold(
        self,
        value
    ):

        self.occlusion_threshold_label.config(
            text=f"{float(value):.2f}"
        )


    # ========================================================
    # DISABLE
    # ========================================================

    def disable_buttons(self):

        def update():

            self.folder_button.config(
                state="disabled"
            )

            self.reference_button_1.config(
                state="disabled"
            )

            self.reference_button_2.config(
                state="disabled"
            )

            self.build_button.config(
                state="disabled"
            )

            self.search_button.config(
                state="disabled"
            )

        self.root.after(
            0,
            update
        )


    # ========================================================
    # ENABLE
    # ========================================================

    def enable_buttons(self):

        def update():

            self.folder_button.config(
                state="normal"
            )

            self.reference_button_1.config(
                state="normal"
            )

            self.reference_button_2.config(
                state="normal"
            )

            self.build_button.config(
                state="normal"
            )

            self.search_button.config(
                state="normal"
            )

        self.root.after(
            0,
            update
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = FacePhotoOrganizer(
        root
    )

    root.mainloop()
