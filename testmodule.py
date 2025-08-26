from sentence_transformers import SentenceTransformer, util  # nlp library
import matplotlib.pyplot as plt   #graph lib
import numpy as np  #for calc

# Loading... the nlp model
model = SentenceTransformer('all-MiniLM-L6-v2')

#--------section cut-----number1
roles = {
    "Cloud Engineer": "Manages cloud platforms like AWS, Azure. Handles VMs, storage, scaling.",
    "DevOps Engineer": "Handles CI/CD, automation, Docker, Kubernetes, infrastructure as code.",
    "Data Engineer": "Builds data pipelines, ETL processes, handles large-scale data architecture.",
    "Machine Learning Engineer": "Builds ML models, training systems, and AI algorithms.",
    "Data Analyst": "Works on dashboards, Excel, Tableau, SQL, interprets datasets.",
    "Web Developer": "Designs and builds websites using HTML, CSS, JS, frontend/backend tools.",
    "Backend Developer": "Focuses on APIs, server logic, DBs using Python, Java, Node.js.",
    "QA/Automation Engineer": "Tests software, writes test scripts using tools like Selenium.",
    "Security Engineer": "Secures applications, networks, cloud infra, handles vulnerabilities."
} #these are roles for industry

irrelevant_keywords = [
    "law", "french", "art", "dance", "history", "geography", "cooking", "economics",
    "politics", "music", "philosophy", "literature", "psychology", "commerce", "language", "accounting", "singing",""
] #a negative type words, or , to not to calculate or a "-1" prompt

# words that indicate advanced level like major role
advanced_keywords = ["Advanced", "Expert", "Professional", "Architect", "Specialist", "Master", "Certified"]

# Recognized institutions or, org that has more value, eg: same certificate from a local org vs aws, may have a diff value
trusted_institutions = [
    "Google", "Microsoft", "Amazon", "AWS", "IBM", "Meta", "KPMG", "TCS", "Infosys",
    "ISRO", "DRDO", "Coursera", "edX", "Udemy", "Udacity"
]

top_companies = [
    "Google", "Amazon", "Microsoft", "ISRO", "DRDO", "KPMG", "EY", "Deloitte", "Infosys", "TCS", "IBM", "Meta","Cisco","PwC","Adobe",
    "ADP","Sony","Dell","Samsung"
]

# ----------------section cut number2--------------------
# input by user
def get_user_input():
    cert_inputs = []
    num_certs = int(input("Enter number of certificates you hold: "))
    print("Enter certificate titles (e.g., Udemy Tableau Certification):")
    for _ in range(num_certs):
        cert_inputs.append(input("- "))

    has_certs = input("Do you have any certifications (like AWS Cloud Practitioner)? (yes/no): ").lower()
    if has_certs == 'yes':
        num_certs2 = int(input("Enter number of certifications: "))
        print("Enter certification names:")
        for _ in range(num_certs2):
            cert_inputs.append(input("- "))

# input for internship details or, training detail if the user done 
    internships = []
    has_intern = input("Do you have any internships/trainings? (yes/no): ").lower()
    if has_intern == 'yes':
        num_intern = int(input("Enter number of internships/trainings: "))
        print("Enter internship/training details:")
        for _ in range(num_intern):
            title = input("  - Role/Title: ")
            company = input("    Company: ")
            duration = int(input("    Duration (in months): "))
            internships.append({"title": title, "company": company, "duration": duration})

    return cert_inputs, internships

# calculation for plotting 
def score_against_roles(cert_list, internships):
    role_scores = {role: 0.0 for role in roles}
    role_embeddings = model.encode(list(roles.values()), convert_to_tensor=True)

# filter irrelevant or negative ones
    filtered_certs = [
        cert for cert in cert_list
        if not any(keyword.lower() in cert.lower() for keyword in irrelevant_keywords)
    ]

# certificate final calculation scoring
    for cert in filtered_certs:
        multiplier = 1.0
        if any(keyword.lower() in cert.lower() for keyword in map(str.lower, advanced_keywords)):
            multiplier += 0.25
        if any(inst.lower() in cert.lower() for inst in map(str.lower, trusted_institutions)):
            multiplier += 0.25

        cert_embedding = model.encode(cert, convert_to_tensor=True)
        cos_scores = util.cos_sim(cert_embedding, role_embeddings)[0]

        for i, role in enumerate(roles):
            role_scores[role] += float(cos_scores[i]) * multiplier

# Internship based calculation
    for intern in internships:
        title = intern["title"]
        company = intern["company"]
        duration = intern["duration"]

        intern_multiplier = 1.0
        if company.strip().lower() in [x.lower() for x in top_companies]:
            intern_multiplier += 0.5
        if duration >= 6:
            intern_multiplier += 0.5
        elif duration >= 3:
            intern_multiplier += 0.25
        elif duration >= 1:
            intern_multiplier += 0.1

        intern_embedding = model.encode(title, convert_to_tensor=True)
        cos_scores = util.cos_sim(intern_embedding, role_embeddings)[0]

        for i, role in enumerate(roles):
            role_scores[role] += float(cos_scores[i]) * intern_multiplier

    return role_scores

# Plotting chart
def plot_radar(role_scores):
    labels = list(role_scores.keys())
    scores = list(role_scores.values())
    max_score = max(scores) if max(scores) != 0 else 1
    normalized = [round((s / max_score) * 10, 2) for s in scores]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    normalized += normalized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.plot(angles, normalized, linewidth=2.5)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, color='black')

    for angle, label, score in zip(angles, labels, normalized):
        ax.text(angle, score + 1.0, f"{score}", color='blue', fontsize=9, ha='center', va='center')

    ax.spines['polar'].set_color('#444')

    plt.title("IT Role Suitability Graph", size=16, fontweight='bold', color='black')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("IT Role Suitability Evaluator (Certifications + Internships)\n")
    cert_list, internships = get_user_input()
    scores = score_against_roles(cert_list, internships)
    plot_radar(scores)
