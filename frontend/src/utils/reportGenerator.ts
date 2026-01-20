import jsPDF from 'jspdf';

const insightLibrary: Record<string, Record<string, string>> = {
  pulpal_depth: {
    Excellent: 'Ideal depth. You successfully bypassed the DEJ into dentin while maintaining a safe distance from the pulp horn.',
    'Clinically Acceptable': 'Deeper than ideal. Use a liner/base (like Dycal or Vitrebond) to protect the pulp from thermal sensitivity and microleakage.',
    'Standard Not Met': 'If too shallow, the restorative material lacks bulk and will fracture. Use your bur head (e.g., #330 is ~1.5-2mm long) as a visual depth gauge.'
  },
  bl_ratio: {
    Excellent: 'Perfect conservation of tooth structure. This width maintains the strength of the buccal and lingual cusps.',
    'Clinically Acceptable': 'The preparation is wide. Be cautious with condensation pressure to avoid stressing the remaining weakened cusps.',
    'Standard Not Met': 'Preparation is too wide. The "wedge effect" during chewing may fracture the cusps. Next time, keep your bur strictly within the central groove.'
  },
  axial_height: {
    Excellent: 'Correct vertical box dimension. Provides adequate resistance form without encroaching on the gingival attachment.',
    'Clinically Acceptable': 'Slightly high or low. Check your gingival floor level—ensure it clears the contact point with the adjacent tooth by 0.5mm.',
    'Standard Not Met': 'If too short, the box won\'t provide enough retention. If too deep, you risk invading "Biological Width." Use a periodontal probe to verify floor depth.'
  },
  wall_taper: {
    Excellent: 'Excellent convergence. This "mechanical lock" prevents the filling from lifting out under sticky food forces.',
    'Clinically Acceptable': 'Near parallel. While acceptable, try to tilt the bur slightly (3-5 degrees) toward the center of the tooth for better retention.',
    'Standard Not Met': 'Divergent walls. The filling will not stay in place. Ensure the handpiece is parallel to the long axis of the tooth, not tilted outward.'
  },
  marginal_ridge: {
    Excellent: 'Strong ridge. This thickness is crucial to prevent the tooth from splitting during heavy occlusal loading.',
    'Clinically Acceptable': 'Borderline thin. Avoid any further distal/mesial extension. Ensure your internal line angles are rounded to reduce stress.',
    'Standard Not Met': 'Ridge is critically weak. In a clinical setting, this ridge would likely fracture. Practice keeping a "safety zone" of tooth structure near the proximal surfaces.'
  },
  wall_smoothness: {
    Excellent: 'Glass-like finish. This ensures the best marginal adaptation and prevents voids at the tooth-material interface.',
    'Clinically Acceptable': 'Slightly rough. Switch to a fine-grit finishing diamond or a multi-fluted carbide bur at low speed for a smoother finish.',
    'Standard Not Met': 'Visible steps or "chatter marks." These create stress concentrations. Use a "planing" motion with a hand instrument (like an enamel hatchet) to smooth the walls.'
  },
  undercuts: {
    Excellent: 'No undercuts. This allows for perfect condensation of composite or amalgam into the line angles.',
    'Clinically Acceptable': 'Minor shadowing. Ensure you are not "burying" the head of the bur into the wall, which creates a bell-shaped prep.',
    'Standard Not Met': 'Significant undercuts found. These trap air bubbles and weaken the enamel. Keep the bur\'s path of motion straight up and down.'
  },
  cusp_undermining: {
    Excellent: 'Cusps are supported by a healthy "dentin foundation." This tooth can withstand high chewing forces.',
    'Clinically Acceptable': 'Dentin support is thin. Be careful when removing decay near the pulp horns to avoid "hollowing out" the cusp.',
    'Standard Not Met': 'The cusp is undermined (enamel with no dentin under it). This cusp will likely break. Consider "capping" the cusp or a more extensive restoration.'
  },
  floor_flatness: {
    Excellent: 'Perfectly flat floor. This "Resistance Form" prevents the restoration from rocking or shifting under pressure.',
    'Clinically Acceptable': 'Slightly uneven. Use a flat-ended bur like a #56 or #245 to gently level the pulpal floor.',
    'Standard Not Met': 'Sloping or "pitted" floor. This creates uneven pressure points. Move the bur in a sweeping, horizontal "mowing" motion rather than pushing down.'
  }
};

export const generateDentalReport = (sessionData: any, metrics: any, screenshot: string) => {
  const doc = new jsPDF('p', 'mm', 'a4');
  const pageWidth = doc.internal.pageSize.width;
  const margin = 14;
  const contentWidth = pageWidth - (margin * 2);

  // === HEADER ===
  doc.setFillColor(251, 251, 253);
  doc.rect(0, 0, pageWidth, 22, 'F');
  
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(17, 17, 19);
  doc.text("Clinical Analysis", margin, 12);
  
  const now = new Date();
  const dateStr = `${String(now.getDate()).padStart(2, '0')}.${String(now.getMonth() + 1).padStart(2, '0')}.${now.getFullYear()}`;
  
  doc.setFontSize(7);
  doc.setTextColor(142, 142, 147);
  doc.text(dateStr, pageWidth - margin, 9, { align: 'right' });
  
  doc.setTextColor(0, 122, 255);
  doc.setFont("helvetica", "bold");
  doc.text("DENTAL AI v2.0", pageWidth - margin, 13, { align: 'right' });
  
  doc.setTextColor(174, 174, 178);
  doc.setFontSize(6);
  doc.text(`File: ${sessionData.filename || 'Untitled'}`, pageWidth - margin, 17, { align: 'right' });

  // === SCAN PREVIEW ===
  const imgY = 28;
  const imgH = 62;
  doc.setDrawColor(229, 229, 234);
  doc.setFillColor(255, 255, 255);
  doc.roundedRect(margin, imgY, contentWidth, imgH, 3, 3, 'FD');

  try {
    const imgProps = doc.getImageProperties(screenshot);
    const maxW = contentWidth - 10;
    const maxH = imgH - 10;
    let imgW = maxW;
    let imgH_calc = (imgProps.height * imgW) / imgProps.width;
    if (imgH_calc > maxH) {
      imgH_calc = maxH;
      imgW = (imgProps.width * imgH_calc) / imgProps.height;
    }
    doc.addImage(screenshot, 'PNG', margin + (contentWidth - imgW) / 2, imgY + (imgH - imgH_calc) / 2, imgW, imgH_calc);
  } catch (e) { console.error(e); }

  // === UNIFIED METRICS GRID (2 Columns) ===
  const sectionY = imgY + imgH + 8;
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(17, 17, 19);
  doc.text("Detailed Metric Analysis", margin, sectionY);
  
  doc.setDrawColor(229, 229, 234);
  doc.line(margin, sectionY + 2.5, pageWidth - margin, sectionY + 2.5);

  const cardW = (contentWidth - 4) / 2;
  const cardH = 30; // Povećano da stane duži tekst saveta
  let cardX = margin;
  let cardY = sectionY + 6;
  let colIndex = 0;

  Object.entries(metrics).forEach(([key, data]: any) => {
    // Provera za novu stranicu
    if (cardY + cardH > 280) {
      doc.addPage();
      cardY = 20;
    }

    // Kartica
    doc.setDrawColor(235, 235, 240);
    doc.setFillColor(255, 255, 255);
    doc.roundedRect(cardX, cardY, cardW, cardH, 2, 2, 'FD');

    // 1. Metrika (Top-Left)
    doc.setFontSize(5.5);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(142, 142, 147);
    const metricName = key.replace(/_/g, ' ').toUpperCase();
    doc.text(metricName, cardX + 4, cardY + 5);

    // 2. Pilula Statusa (Top-Right)
    let textColor = [99, 99, 102], bgColor = [242, 242, 247];
    let statusLabel = 'UNKNOWN';
    if (data.grade.includes('Excellent')) { 
      textColor = [48, 176, 80]; bgColor = [237, 247, 237]; statusLabel = 'EXCELLENT';
    } else if (data.grade.includes('Acceptable')) { 
      textColor = [245, 139, 0]; bgColor = [255, 250, 235]; statusLabel = 'ACCEPTABLE';
    } else if (data.grade.includes('Not Met')) { 
      textColor = [245, 49, 38]; bgColor = [255, 239, 239]; statusLabel = 'NOT MET';
    }

    doc.setFontSize(5);
    const badgeW = doc.getTextWidth(statusLabel) + 4;
    const badgeX = (cardX + cardW) - badgeW - 4;
    doc.setFillColor(bgColor[0], bgColor[1], bgColor[2]);
    doc.roundedRect(badgeX, cardY + 2.5, badgeW, 3.5, 1, 1, 'F');
    doc.setTextColor(textColor[0], textColor[1], textColor[2]);
    doc.text(statusLabel, badgeX + badgeW / 2, cardY + 4.8, { align: 'center' });

    // 3. Vrednost
    doc.setFontSize(11);
    doc.setTextColor(17, 17, 19);
    doc.setFont("helvetica", "bold");
    const valueStr = `${data.value ?? 'N/A'}${data.unit ? ' ' + data.unit : ''}`;
    doc.text(valueStr, cardX + 4, cardY + 11.5);

    // 4. Linija razdelnica
    doc.setDrawColor(242, 242, 247);
    doc.line(cardX + 4, cardY + 14, cardX + cardW - 4, cardY + 14);

    // 5. Klinički savet (Insight)
    const adviceText = insightLibrary[key]?.[data.grade] || "Guidance not available for this metric.";
    doc.setFontSize(6.2);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(60, 60, 67);
    const wrappedAdvice = doc.splitTextToSize(adviceText, cardW - 8);
    doc.text(wrappedAdvice, cardX + 4, cardY + 18.5);

    // Grid navigacija
    colIndex++;
    if (colIndex === 2) {
      colIndex = 0;
      cardX = margin;
      cardY += cardH + 3;
    } else {
      cardX += cardW + 4;
    }
  });

  // === FOOTER ===
  doc.setFontSize(6);
  doc.setTextColor(174, 174, 178);
  doc.text("DENTAL AI ENGINE • EDUCATIONAL ANALYSIS", pageWidth / 2, 290, { align: 'center' });

  doc.save(`DentalReport_${sessionData.filename?.split('.')[0] || 'Analysis'}.pdf`);
};