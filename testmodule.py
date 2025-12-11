from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------- ROLE DEFINITIONS --------------------
roles = {
    "Cloud Engineer": "Manages cloud platforms like AWS, Azure.",
    "DevOps Engineer": "Handles CI/CD, Docker, Kubernetes.",
    "Data Engineer": "Builds ETL pipelines and data architecture.",
    "Machine Learning Engineer": "Builds ML models and AI systems.",
    "Data Analyst": "Works on Tableau, Excel, SQL dashboards.",
    "Web Developer": "Builds websites using HTML, CSS, JS.",
    "Backend Developer": "APIs, server logic using Python/Java.",
    "QA/Automation Engineer": "Software testing, automation tools.",
    "Security Engineer": "Secures networks and applications."
}

irrelevant_keywords = [
    "law", "french", "art", "dance", "history", "geography", "cooking",
    "politics", "music", "philosophy", "commerce", "language", "accounting"
]

advanced_keywords = ["Advanced", "Expert", "Professional", "Architect", "Specialist", "Master", "Certified"]

trusted_institutions = [
    "Google", "Microsoft", "Amazon", "AWS", "IBM", "Meta", "KPMG", "TCS",
    "Infosys", "ISRO", "DRDO", "Coursera", "edX", "Udemy", "Udacity"
]

top_companies = [
    "Google", "Amazon", "Microsoft", "ISRO", "DRDO", "KPMG", "EY", "Deloitte",
    "Infosys", "TCS", "IBM", "Meta", "Cisco", "PwC", "Adobe", "Sony", "Dell"
]

# ---------------------- NORMALIZATION ------------------------
def min_max_norm(score, min_val=0, max_val=10):
    return (score - min_val) / (max_val - min_val)

def softmax(values):
    e = np.exp(values)
    return e / np.sum(e)

# ---------------------- USER INPUT ----------------------------
def get_user_input():
    certs = []
    num_certs = int(input("Enter number of certificates: "))
    print("Enter certificate titles:")
    for _ in range(num_certs):
        certs.append(input("- "))

    # Internship Section
    internships = []
    if input("Any internships? (yes/no): ").lower() == "yes":
        count = int(input("How many internships? "))
        for _ in range(count):
            title = input("  Role: ")
            company = input("  Company: ")
            duration = int(input("  Duration (months): "))
            internships.append({"title": title, "company": company, "duration": duration})

    # Projects Section
    projects = []
    if input("Any projects? (yes/no): ").lower() == "yes":
        count = int(input("How many projects? "))
        for _ in range(count):
            p_title = input("  Project Title: ")
            p_desc = input("  Short Description: ")
            p_level = input("  Level (basic/intermediate/advanced): ").lower()
            projects.append({"title": p_title, "desc": p_desc, "level": p_level})

    return certs, internships, projects

# ---------------------- SCORING ENGINE ------------------------
def score_against_roles(certs, internships, projects):
    role_scores = {role: 0.0 for role in roles}
    role_embeddings = model.encode(list(roles.values()), convert_to_tensor=True)

    # ---------------- 1. CERTIFICATE SCORING (10%) -----------------
    filtered = [
        c for c in certs
        if not any(bad in c.lower() for bad in irrelevant_keywords)
    ]

    cert_strength = []

    for cert in filtered:
        multiplier = 1.0

        # Advanced level boost
        if any(k.lower() in cert.lower() for k in advanced_keywords):
            multiplier += 0.25

        # Trusted institution boost
        if any(inst.lower() in cert.lower() for inst in map(str.lower, trusted_institutions)):
            multiplier += 0.25

        cert_embed = model.encode(cert, convert_to_tensor=True)
        cos_scores = util.cos_sim(cert_embed, role_embeddings)[0]

        cert_strength.append([float(x) * multiplier for x in cos_scores])

    if cert_strength:
        cert_strength = np.mean(cert_strength, axis=0)
        cert_strength = softmax(cert_strength) * 10
    else:
        cert_strength = [0] * len(roles)

    # ---------------- 2. INTERNSHIP SCORING (60%) -----------------
    intern_strength = []

    for intern in internships:
        multiplier = 1.0

        if intern["company"].lower() in [c.lower() for c in top_companies]:
            multiplier += 0.5

        dur = intern["duration"]
        if dur >= 6:
            multiplier += 0.5
        elif dur >= 3:
            multiplier += 0.25
        else:
            multiplier += 0.1

        embed = model.encode(intern["title"], convert_to_tensor=True)
        cos_scores = util.cos_sim(embed, role_embeddings)[0]

        intern_strength.append([float(x) * multiplier for x in cos_scores])

    if intern_strength:
        intern_strength = np.mean(intern_strength, axis=0)
        intern_strength = softmax(intern_strength) * 10
    else:
        intern_strength = [0] * len(roles)

    # ---------------- 3. PROJECT SCORING (30%) -----------------
    proj_strength = []

    for proj in projects:
        multiplier = 1.0
        if proj["level"] == "advanced":
            multiplier += 0.3
        elif proj["level"] == "intermediate":
            multiplier += 0.15

        text = proj["title"] + " " + proj["desc"]
        embed = model.encode(text, convert_to_tensor=True)
        cos_scores = util.cos_sim(embed, role_embeddings)[0]
        proj_strength.append([float(x) * multiplier for x in cos_scores])

    if proj_strength:
        proj_strength = np.mean(proj_strength, axis=0)
        proj_strength = softmax(proj_strength) * 10
    else:
        proj_strength = [0] * len(roles)

    # ------------------ FINAL WEIGHTED SCORE ----------------------
    final_scores = {}
    role_list = list(roles.keys())

    for i, role in enumerate(role_list):
        final_scores[role] = (
            intern_strength[i] * 0.60 +
            proj_strength[i] * 0.30 +
            cert_strength[i] * 0.10
        )

    return final_scores

# -------------------- RADAR PLOT ---------------------------
def plot_radar(role_scores):
    labels = list(role_scores.keys())
    values = list(role_scores.values())

    max_val = max(values) if max(values) != 0 else 1
    normalized = [round((v / max_val) * 10, 2) for v in values]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    normalized += normalized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, normalized, linewidth=2.5)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    for angle, score in zip(angles, normalized):
        ax.text(angle, score + 1, str(score), ha='center')

    plt.title("IT Role Suitability Chart (Cert + Internship + Projects)", fontsize=15, fontweight="bold")
    plt.show()

# ---------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("\n--- IT Role Suitability AI Model ---\n")
    certs, interns, projects = get_user_input()
    scores = score_against_roles(certs, interns, projects)
    plot_radar(scores)
    
