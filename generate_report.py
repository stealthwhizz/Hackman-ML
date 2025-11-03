"""
Generate Hackathon Project Report PDF
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.colors import HexColor
from datetime import datetime

def create_hackathon_report():
    """Create comprehensive Hackathon project report PDF"""
    
    # Create PDF
    pdf_filename = "Hackathon_Project_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HexColor('#34495e'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HexColor('#2980b9'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=HexColor('#27ae60'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=HexColor('#2c3e50'),
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # ============= PAGE 1: TITLE PAGE =============
    elements.append(Spacer(1, 1.5*inch))
    
    # Title
    title = Paragraph("INTELLIGENT HANGMAN AGENT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Subtitle
    subtitle = Paragraph(
        "A Hybrid Machine Learning Approach using<br/>Hidden Markov Models and Deep Q-Networks",
        subtitle_style
    )
    elements.append(subtitle)
    elements.append(Spacer(1, 0.5*inch))
    
    # Project Info Box
    project_info = [
        ['<b>Project Type:</b>', 'Hackathon Submission'],
        ['<b>Repository:</b>', 'Hackman-ML'],
        ['<b>Author:</b>', 'Akshay AG'],
        ['<b>Date:</b>', datetime.now().strftime('%B %d, %Y')],
        ['<b>Final Accuracy:</b>', '<b><font color="#27ae60">32.55%</font></b>'],
    ]
    
    info_table = Table(project_info, colWidths=[2*inch, 3.5*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # Abstract
    abstract_title = Paragraph("<b>ABSTRACT</b>", heading_style)
    elements.append(abstract_title)
    
    abstract_text = """
    This project presents an intelligent agent for playing the game of Hangman using advanced 
    machine learning techniques. We developed and compared multiple approaches including Hidden 
    Markov Models (HMM), Deep Q-Networks (DQN), Imitation Learning, and Ensemble methods. 
    The final system achieves a <b>32.55% win rate</b> on a test dataset of 2000 words, 
    demonstrating the effectiveness of statistical pattern matching combined with reinforcement 
    learning for strategic word guessing games.
    """
    elements.append(Paragraph(abstract_text, body_style))
    
    elements.append(PageBreak())
    
    # ============= PAGE 2: INTRODUCTION & PROBLEM STATEMENT =============
    
    # Introduction
    intro_title = Paragraph("1. INTRODUCTION", heading_style)
    elements.append(intro_title)
    
    intro_text = """
    Hangman is a classic word-guessing game where players attempt to identify a hidden word 
    by suggesting letters within a limited number of guesses. Each incorrect guess results in 
    a penalty, and the game ends when the word is revealed or the maximum number of wrong guesses 
    (typically 6) is reached. This project explores the application of artificial intelligence 
    and machine learning to create an intelligent agent capable of playing Hangman strategically.
    """
    elements.append(Paragraph(intro_text, body_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Problem Statement
    problem_title = Paragraph("2. PROBLEM STATEMENT", heading_style)
    elements.append(problem_title)
    
    problem_text = """
    The challenge is to develop an AI agent that can:
    <br/>• Maximize the win rate by guessing words correctly within 6 wrong attempts
    <br/>• Learn optimal letter selection strategies from a corpus of 50,000 words
    <br/>• Generalize well to unseen words in a test set
    <br/>• Balance exploration (trying new letters) with exploitation (using learned patterns)
    <br/>• Handle words of varying lengths and complexity
    """
    elements.append(Paragraph(problem_text, body_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Objectives
    objectives_title = Paragraph("2.1 Objectives", subheading_style)
    elements.append(objectives_title)
    
    objectives = [
        ['1.', 'Implement and compare multiple ML approaches for Hangman'],
        ['2.', 'Achieve competitive performance on standard test benchmarks'],
        ['3.', 'Develop an interactive GUI for human-AI gameplay'],
        ['4.', 'Analyze the strengths and weaknesses of each approach'],
        ['5.', 'Create a production-ready, well-documented system'],
    ]
    
    obj_table = Table(objectives, colWidths=[0.3*inch, 5.2*inch])
    obj_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#2c3e50')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(obj_table)
    
    elements.append(PageBreak())
    
    # ============= PAGE 3: METHODOLOGY =============
    
    method_title = Paragraph("3. METHODOLOGY", heading_style)
    elements.append(method_title)
    
    # 3.1 Data Collection
    data_title = Paragraph("3.1 Data Collection and Preparation", subheading_style)
    elements.append(data_title)
    
    data_text = """
    <b>Training Dataset:</b> 50,000 English words from a comprehensive corpus
    <br/><b>Test Dataset:</b> 2,000 randomly selected words (seed=42) for consistent evaluation
    <br/><b>Word Length Range:</b> 1-19 characters
    <br/><b>Preprocessing:</b> Lowercase conversion, whitespace trimming, validation
    """
    elements.append(Paragraph(data_text, body_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # 3.2 Approaches Implemented
    approaches_title = Paragraph("3.2 Approaches Implemented", subheading_style)
    elements.append(approaches_title)
    
    # Approach 1: HMM
    hmm_title = Paragraph("<b>3.2.1 Hidden Markov Model (HMM)</b>", body_style)
    elements.append(hmm_title)
    
    hmm_text = """
    The HMM approach uses statistical pattern matching to predict the most likely letters:
    <br/>• <b>Separate Models:</b> One model trained for each word length (1-19 characters)
    <br/>• <b>Letter Position Frequencies:</b> Tracks where each letter appears in words
    <br/>• <b>Bigram Patterns:</b> Analyzes two-letter combinations for context
    <br/>• <b>Probability Calculation:</b> Computes likelihood of each letter given current masked word
    <br/>• <b>Training Time:</b> Less than 3 seconds (instant)
    <br/>• <b>Advantages:</b> Fast, interpretable, no neural network required
    """
    elements.append(Paragraph(hmm_text, body_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Approach 2: DQN
    dqn_title = Paragraph("<b>3.2.2 Deep Q-Network (DQN)</b>", body_style)
    elements.append(dqn_title)
    
    dqn_text = """
    A reinforcement learning approach using deep neural networks:
    <br/>• <b>Architecture:</b> 3-layer neural network (Input → 256 → 128 → 26 outputs)
    <br/>• <b>State Representation:</b> Masked word, guessed letters, lives remaining, HMM probabilities
    <br/>• <b>Action Space:</b> 26 possible letters (a-z)
    <br/>• <b>Reward System:</b> +10 for correct, -1 for wrong, +100 for winning
    <br/>• <b>Training:</b> Experience replay with ε-greedy exploration
    <br/>• <b>Episodes:</b> 10,000-50,000 training games
    <br/>• <b>Advantages:</b> Can learn complex strategies, adapts to game state
    """
    elements.append(Paragraph(dqn_text, body_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Approach 3: Imitation Learning
    imitation_title = Paragraph("<b>3.2.3 Imitation Learning</b>", body_style)
    elements.append(imitation_title)
    
    imitation_text = """
    Learns from expert demonstrations (HMM agent):
    <br/>• Collects 500 demonstration games from HMM agent
    <br/>• Trains neural network to mimic expert behavior
    <br/>• Uses supervised learning on (state, action) pairs
    <br/>• Training Time: ~45 seconds
    """
    elements.append(Paragraph(imitation_text, body_style))
    
    elements.append(PageBreak())
    
    # ============= PAGE 4: RESULTS =============
    
    results_title = Paragraph("4. RESULTS AND ANALYSIS", heading_style)
    elements.append(results_title)
    
    results_overview = Paragraph("4.1 Performance Comparison", subheading_style)
    elements.append(results_overview)
    
    # Results table
    results_data = [
        ['<b>Approach</b>', '<b>Win Rate</b>', '<b>Score</b>', '<b>Training Time</b>', '<b>Rank</b>'],
        ['Pure HMM', '<b><font color="#27ae60">32.55%</font></b>', '-39,915', '< 3 seconds', '🥇 1st'],
        ['Imitation Learning', '8.8%', '-54,475', '45 seconds', '🥈 2nd'],
        ['Fast Transfer DQN', '6.0%', '-57,955', '1.9 minutes', '🥉 3rd'],
        ['Ensemble (HMM+DQN)', '3.75%', '-57,955', '2.0 minutes', '4th'],
    ]
    
    results_table = Table(results_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.2*inch, 0.8*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor('#d5f4e6')),
        ('BACKGROUND', (0, 2), (-1, -1), HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2c3e50')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#95a5a6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8f9fa')]),
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 0.15*inch))
    
    # Key Findings
    findings_title = Paragraph("4.2 Key Findings", subheading_style)
    elements.append(findings_title)
    
    findings_text = """
    <b>Winner: Pure HMM with 32.55% Win Rate</b>
    <br/><br/>
    The statistical Hidden Markov Model approach emerged as the clear winner, outperforming 
    all neural network-based methods. This demonstrates that for word-guessing games with 
    rich pattern data:
    <br/>• <b>Domain Knowledge Wins:</b> Statistical pattern matching leveraging 50,000 word 
    corpus beats complex neural networks
    <br/>• <b>Training Efficiency:</b> HMM requires seconds while DQN needs hours of training
    <br/>• <b>Interpretability:</b> HMM decisions are transparent and explainable
    <br/>• <b>Generalization:</b> Strong performance on unseen test words
    <br/><br/>
    <b>Why Neural Networks Struggled:</b>
    <br/>• Limited training time (< 10,000 episodes vs. 50,000+ needed)
    <br/>• Sparse reward signals in Hangman
    <br/>• High-dimensional state space (26 letters × word positions)
    <br/>• HMM already captures most useful patterns
    """
    elements.append(Paragraph(findings_text, body_style))
    
    elements.append(PageBreak())
    
    # ============= PAGE 5: TECHNICAL IMPLEMENTATION =============
    
    tech_title = Paragraph("5. TECHNICAL IMPLEMENTATION", heading_style)
    elements.append(tech_title)
    
    # Architecture
    arch_title = Paragraph("5.1 System Architecture", subheading_style)
    elements.append(arch_title)
    
    arch_text = """
    The system consists of modular components:
    <br/>• <b>Data Layer:</b> Corpus and test word management
    <br/>• <b>Model Layer:</b> HMM and DQN implementations
    <br/>• <b>Environment Layer:</b> Game simulation and state management
    <br/>• <b>Training Layer:</b> Scripts for model training and optimization
    <br/>• <b>Interface Layer:</b> Streamlit GUI for interactive gameplay
    """
    elements.append(Paragraph(arch_text, body_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Technologies
    tech_stack_title = Paragraph("5.2 Technology Stack", subheading_style)
    elements.append(tech_stack_title)
    
    tech_data = [
        ['<b>Category</b>', '<b>Technology</b>', '<b>Purpose</b>'],
        ['Language', 'Python 3.12+', 'Core implementation'],
        ['Deep Learning', 'PyTorch', 'Neural network training'],
        ['GUI Framework', 'Streamlit', 'Interactive interface'],
        ['Data Processing', 'NumPy, Pandas', 'Data manipulation'],
        ['Visualization', 'Matplotlib', 'Charts and diagrams'],
        ['Version Control', 'Git, GitHub', 'Code management'],
    ]
    
    tech_table = Table(tech_data, colWidths=[1.5*inch, 1.8*inch, 2.2*inch])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2c3e50')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(tech_table)
    elements.append(Spacer(1, 0.15*inch))
    
    # Features
    features_title = Paragraph("5.3 Key Features", subheading_style)
    elements.append(features_title)
    
    features_text = """
    <b>Interactive GUI:</b>
    <br/>• Human vs AI gameplay
    <br/>• AI auto-play mode (watch AI play automatically)
    <br/>• Real-time statistics and visualizations
    <br/>• Multiple difficulty levels
    <br/>• Model selection (HMM or DQN)
    <br/><br/>
    <b>Analysis Tools:</b>
    <br/>• Comprehensive performance metrics
    <br/>• Win/loss distribution charts
    <br/>• Word length analysis
    <br/>• Cumulative score tracking
    <br/>• Detailed game logs and reports
    """
    elements.append(Paragraph(features_text, body_style))
    
    elements.append(PageBreak())
    
    # ============= PAGE 6: CHALLENGES & FUTURE WORK =============
    
    challenges_title = Paragraph("6. CHALLENGES AND LESSONS LEARNED", heading_style)
    elements.append(challenges_title)
    
    challenges_text = """
    <b>Technical Challenges:</b>
    <br/>• <b>DQN Training Instability:</b> Required careful hyperparameter tuning and 
    curriculum learning to achieve stable training
    <br/>• <b>Sparse Rewards:</b> Most game actions provide minimal feedback, making RL challenging
    <br/>• <b>High Variance:</b> Hangman has inherent randomness in word selection
    <br/>• <b>Computational Cost:</b> Neural network training requires significant time (2-3 hours)
    <br/><br/>
    <b>Key Lessons:</b>
    <br/>• Simple statistical methods can outperform complex neural networks when patterns 
    are well-defined
    <br/>• Domain knowledge (letter frequencies, word patterns) is invaluable
    <br/>• Training efficiency matters - HMM's instant training vs DQN's hours
    <br/>• Proper evaluation on held-out test sets is crucial for fair comparison
    """
    elements.append(Paragraph(challenges_text, body_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Future Work
    future_title = Paragraph("7. FUTURE WORK AND IMPROVEMENTS", heading_style)
    elements.append(future_title)
    
    future_text = """
    <b>Potential Enhancements:</b>
    <br/>• <b>Advanced DQN Variants:</b> Implement Double DQN, Dueling DQN, or Rainbow 
    for better performance
    <br/>• <b>Transfer Learning:</b> Pre-train on multiple languages or word games
    <br/>• <b>Attention Mechanisms:</b> Focus on important word positions and patterns
    <br/>• <b>Meta-Learning:</b> Learn to adapt quickly to new word distributions
    <br/>• <b>Explainable AI:</b> Visualize what patterns the models learn
    <br/>• <b>Multi-Modal Learning:</b> Combine multiple information sources
    <br/>• <b>Online Learning:</b> Continuously improve from gameplay experience
    <br/><br/>
    <b>Deployment Ideas:</b>
    <br/>• Web application for public access
    <br/>• Mobile app for on-the-go gameplay
    <br/>• Educational tool for teaching AI concepts
    <br/>• Benchmark for comparing AI agents
    """
    elements.append(Paragraph(future_text, body_style))
    
    elements.append(PageBreak())
    
    # ============= PAGE 7: CONCLUSION =============
    
    conclusion_title = Paragraph("8. CONCLUSION", heading_style)
    elements.append(conclusion_title)
    
    conclusion_text = """
    This hackathon project successfully demonstrates the application of machine learning 
    to the classic Hangman game. Through rigorous experimentation with multiple approaches 
    (HMM, DQN, Imitation Learning, and Ensemble methods), we achieved a <b>final accuracy 
    of 32.55%</b> using the Hidden Markov Model approach.
    <br/><br/>
    <b>Key Achievements:</b>
    <br/>✓ Implemented and compared 4 distinct ML approaches
    <br/>✓ Achieved 32.55% win rate (651 wins out of 2000 test games)
    <br/>✓ Developed production-ready code with clean architecture
    <br/>✓ Created interactive GUI for human-AI gameplay
    <br/>✓ Generated comprehensive analysis and visualizations
    <br/>✓ Documented entire process in Jupyter notebook
    <br/><br/>
    <b>Impact and Significance:</b>
    <br/>
    The project demonstrates that classical statistical methods, when properly applied 
    to structured domains with rich pattern data, can outperform modern deep learning 
    approaches. This has important implications for:
    <br/>• Resource-constrained applications where training time matters
    <br/>• Problems where interpretability is crucial
    <br/>• Domains with well-defined patterns and limited training data
    <br/><br/>
    The 32.55% win rate, while not perfect, represents a strong performance given the 
    difficulty of the task. Random guessing would achieve less than 5% win rate, and 
    frequency-only approaches typically max out around 20%. Our HMM agent's performance 
    demonstrates sophisticated pattern recognition and strategic decision-making.
    """
    elements.append(Paragraph(conclusion_text, body_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Final Statistics Box
    final_stats = [
        ['<b>Final Performance Summary</b>', ''],
        ['Win Rate', '<b><font color="#27ae60">32.55%</font></b>'],
        ['Total Games Tested', '2,000'],
        ['Games Won', '651'],
        ['Games Lost', '1,349'],
        ['Average Wrong Guesses', '4.8 per game'],
        ['Total Score', '-39,915'],
        ['Best Approach', 'Hidden Markov Model'],
    ]
    
    stats_table = Table(final_stats, colWidths=[3*inch, 2.5*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('SPAN', (0, 0), (-1, 0)),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2c3e50')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#95a5a6')),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
    ]))
    elements.append(stats_table)
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Repository Information
    repo_title = Paragraph("9. REPOSITORY AND RESOURCES", heading_style)
    elements.append(repo_title)
    
    repo_text = """
    <b>GitHub Repository:</b> github.com/akshayag2005/Hackman-ML
    <br/><b>Project Structure:</b>
    <br/>• <b>src/</b> - All source code (models, training, GUI)
    <br/>• <b>Data/</b> - Training corpus and test datasets
    <br/>• <b>models/</b> - Saved model weights
    <br/>• <b>Assets/</b> - Generated visualizations and diagrams
    <br/>• <b>hangman_agent.ipynb</b> - Complete implementation notebook
    <br/><br/>
    <b>How to Run:</b>
    <br/>1. Install dependencies: <font face="Courier">pip install -r requirements.txt</font>
    <br/>2. Launch GUI: <font face="Courier">streamlit run src/hangman_gui.py</font>
    <br/>3. Train models: <font face="Courier">python src/train_quick_dqn.py</font>
    <br/>4. Run tests: <font face="Courier">python src/test_model.py</font>
    """
    elements.append(Paragraph(repo_text, body_style))
    
    elements.append(Spacer(1, 0.5*inch))
    
    # Footer
    footer_text = """
    <br/><br/>
    ────────────────────────────────────────────────────────────────
    <br/><br/>
    <b>End of Report</b>
    <br/>
    This report was generated on <b>{}</b>
    <br/>
    For questions or collaboration, please contact via GitHub.
    """.format(datetime.now().strftime('%B %d, %Y at %I:%M %p'))
    
    elements.append(Paragraph(footer_text, ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=HexColor('#7f8c8d'),
        alignment=TA_CENTER
    )))
    
    # Build PDF
    doc.build(elements)
    
    return pdf_filename

if __name__ == "__main__":
    print("Generating Hackathon Project Report PDF...")
    filename = create_hackathon_report()
    print(f"✓ Report generated successfully: {filename}")
    print(f"✓ Final accuracy: 32.55%")
