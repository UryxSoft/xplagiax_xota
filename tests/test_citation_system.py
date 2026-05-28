"""
Tests for the citation detection system.
Run with: python -m pytest tests/test_citation_system.py -v
"""

import pytest
from app.antiplagio.citation.detector import CitationDetector, CitationStyle, ZoneType


# ─────────────────────────────────────────────
# Sample texts
# ─────────────────────────────────────────────

TEXT_APA_COMPLETO = """
La inteligencia artificial ha transformado radicalmente los métodos de enseñanza
en la educación superior (García & López, 2021). Según Smith et al. (2020), el
aprendizaje automático permite personalizar el contenido educativo de manera
efectiva. Como señala Johnson (2019, p. 45): "La adaptación del currículo mediante
algoritmos reduce en un 40% el tiempo de aprendizaje en estudiencias universitarias".

Los sistemas de tutoría inteligente han demostrado ser altamente eficaces en
contextos latinoamericanos (Rodríguez, 2022; Martínez & Chen, 2021). Sin embargo,
Pérez (2023, pp. 112-115) argumenta que la brecha tecnológica limita su adopción.

Referencias

García, M., & López, J. (2021). Transformación digital en universidades latinoamericanas.
    Revista de Educación Superior, 15(3), 45-67. https://doi.org/10.1234/res.2021.003

Smith, A., Brown, B., & White, C. (2020). Machine learning in higher education.
    Journal of Educational Technology, 8(2), 123-145.

Johnson, R. (2019). Adaptive learning systems: A comprehensive review.
    Oxford University Press.

Rodríguez, P. (2022). IA en universidades de América Latina. Tecnología Educativa, 5(1), 20-35.

Martínez, L., & Chen, X. (2021). Cross-cultural applications of ITS.
    Educational Informatics, 12(4), 89-104.

Pérez, A. (2023). Barreras tecnológicas en la adopción de IA educativa.
    Tesis doctoral. Universidad Nacional Autónoma de México.
"""

TEXT_IEEE_COMPLETO = """
Recent advances in transformer architectures have significantly improved natural
language processing capabilities [1]. The attention mechanism, first introduced
by Vaswani et al., enables models to process long-range dependencies [2,3].
Performance benchmarks show accuracy improvements of up to 35% compared to
previous methods [4].

The BERT model [5] and its variants have become standard baselines for most NLP
tasks. Fine-tuning pre-trained models requires substantially less labeled data [1,6].

References
[1] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, vol. 521, pp. 436-444, 2015.
[2] A. Vaswani et al., "Attention is all you need," in Proc. NeurIPS, pp. 5998-6008, 2017.
[3] J. Devlin et al., "BERT: Pre-training of deep bidirectional transformers," in Proc. NAACL, 2019.
[4] T. Brown et al., "Language models are few-shot learners," NeurIPS, vol. 33, 2020.
[5] J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT," arXiv:1810.04805, 2018.
[6] P. Liu et al., "Pre-train, prompt, and predict," ACM Comput. Surv., 2023.
"""

TEXT_PLAGIO_SIN_CITAS = """
La inteligencia artificial ha transformado radicalmente los métodos de enseñanza
en la educación superior. El aprendizaje automático permite personalizar el
contenido educativo de manera efectiva. La adaptación del currículo mediante
algoritmos reduce en un 40% el tiempo de aprendizaje en estudiantes universitarios.

Los sistemas de tutoría inteligente han demostrado ser altamente eficaces en
contextos latinoamericanos. Sin embargo, la brecha tecnológica limita su adopción
en países en desarrollo, especialmente cuando los recursos computacionales son
escasos y la infraestructura digital es deficiente.
"""

TEXT_ESTILO_MIXTO = """
La globalización económica ha generado desigualdades estructurales (Piketty, 2014).
World Bank [3] data shows that income inequality increased 15% in developing nations.
Harvey (2005) argues that neoliberalism restructures social relations fundamentally.
Rodrik (2011, pp. 200-215) presents empirical evidence for these claims.

References
[3] World Bank. (2020). World Development Report 2020. Washington DC.
Piketty, T. (2014). Capital in the Twenty-First Century. Harvard University Press.
Harvey, D. (2005). A brief history of neoliberalism. Oxford University Press.
"""


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestCitationDetection:

    def setup_method(self):
        self.detector = CitationDetector()

    def test_apa_inline_detection(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        apa_cites = [c for c in result.inline_citations if c.style == CitationStyle.APA]
        assert len(apa_cites) >= 5, f"Expected ≥5 APA citations, found {len(apa_cites)}"

    def test_apa_with_pages(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        paged = [c for c in result.inline_citations if c.page is not None]
        assert len(paged) >= 2, f"Expected ≥2 citations with page, found {len(paged)}"

    def test_apa_narrative_form(self):
        text = "Como señala Johnson (2019), los sistemas adaptativos funcionan bien."
        result = self.detector.analyze(text)
        assert len(result.inline_citations) >= 1
        assert result.inline_citations[0].author is not None

    def test_ieee_detection(self):
        result = self.detector.analyze(TEXT_IEEE_COMPLETO)
        ieee_cites = [c for c in result.inline_citations if c.style == CitationStyle.IEEE]
        assert len(ieee_cites) >= 4, f"Expected ≥4 IEEE citations, found {len(ieee_cites)}"

    def test_bibliography_parsing_apa(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        assert len(result.bibliography) >= 5, (
            f"Expected ≥5 bibliography entries, found {len(result.bibliography)}"
        )

    def test_bibliography_parsing_ieee(self):
        result = self.detector.analyze(TEXT_IEEE_COMPLETO)
        assert len(result.bibliography) >= 4

    def test_cross_linking(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        orphan_ratio = len(result.orphan_citations) / max(len(result.inline_citations), 1)
        assert orphan_ratio < 0.5, (
            f"Too many orphan citations: {len(result.orphan_citations)}/{len(result.inline_citations)}"
        )

    def test_dominant_style_apa(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        assert result.dominant_style == CitationStyle.APA

    def test_dominant_style_ieee(self):
        result = self.detector.analyze(TEXT_IEEE_COMPLETO)
        assert result.dominant_style == CitationStyle.IEEE

    def test_direct_quote_zone(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        quote_zones = [z for z in result.zones if z.zone_type == ZoneType.DIRECT_QUOTE]
        assert len(quote_zones) >= 1, "Should detect at least one direct quote"

    def test_bibliography_zone(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        bib_zones = [z for z in result.zones if z.zone_type == ZoneType.BIBLIOGRAPHY]
        assert len(bib_zones) == 1, "Should have exactly 1 bibliography zone"

    def test_style_consistency_pure_apa(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        assert result.style_consistency >= 0.70, (
            f"Low consistency for pure APA text: {result.style_consistency:.0%}"
        )

    def test_mixed_style_detection(self):
        result = self.detector.analyze(TEXT_ESTILO_MIXTO)
        assert result.style_consistency < 0.85, (
            f"Should detect style inconsistency, consistency={result.style_consistency:.0%}"
        )


class TestPlagiarismRisk:

    def setup_method(self):
        self.detector = CitationDetector()

    def test_uncited_text_high_risk(self):
        result = self.detector.analyze(TEXT_PLAGIO_SIN_CITAS)
        original_zones = [z for z in result.zones if z.zone_type == ZoneType.ORIGINAL]
        for zone in original_zones:
            assert not zone.has_valid_citation

    def test_properly_cited_text_has_cited_zones(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        cited_zones = [z for z in result.zones if z.has_valid_citation]
        assert len(cited_zones) >= 1, "Should have zones with valid citations"

    def test_citation_coverage_metric(self):
        result_cited = self.detector.analyze(TEXT_APA_COMPLETO)
        assert isinstance(result_cited.citation_coverage, float)
        assert 0.0 <= result_cited.citation_coverage <= 1.0

    def test_doi_extraction(self):
        result = self.detector.analyze(TEXT_APA_COMPLETO)
        doi_entries = [e for e in result.bibliography if e.doi]
        assert len(doi_entries) >= 1, "Should extract at least one DOI"
        assert "10." in doi_entries[0].doi


class TestEdgeCases:

    def setup_method(self):
        self.detector = CitationDetector()

    def test_empty_text(self):
        result = self.detector.analyze("")
        assert result.dominant_style == CitationStyle.UNKNOWN
        assert len(result.inline_citations) == 0

    def test_text_without_citations(self):
        result = self.detector.analyze("Este es un texto sin ninguna referencia bibliográfica.")
        assert len(result.inline_citations) == 0
        assert len(result.bibliography) == 0
        assert result.citation_coverage == 1.0

    def test_spanish_author_names(self):
        text = "Según Martínez-Peñaloza & García (2022), los resultados son significativos."
        result = self.detector.analyze(text)
        assert len(result.inline_citations) >= 1

    def test_et_al_citation(self):
        text = "Como demuestran Brown et al. (2021), este fenómeno es universal."
        result = self.detector.analyze(text)
        assert len(result.inline_citations) >= 1
        assert result.inline_citations[0].author is not None

    def test_multiple_citations_same_parenthesis(self):
        text = "Esto ha sido ampliamente documentado (García, 2021; López, 2020; Martínez, 2019)."
        result = self.detector.analyze(text)
        assert len(result.inline_citations) >= 2

    def test_block_quote_detection(self):
        text = """
El siguiente párrafo fue extraído directamente:

    La transformación digital requiere no solo inversión tecnológica
    sino también un cambio cultural profundo en las organizaciones.

Esto confirma la tesis central del documento.
"""
        result = self.detector.analyze(text)
        assert result is not None


if __name__ == "__main__":
    detector = CitationDetector()

    print("=" * 60)
    print("TEST: Texto APA completo")
    print("=" * 60)
    result = detector.analyze(TEXT_APA_COMPLETO)
    print(f"  Estilo dominante: {result.dominant_style.value}")
    print(f"  Citas inline: {len(result.inline_citations)}")
    print(f"  Entradas bibliográficas: {len(result.bibliography)}")
    print(f"  Citas huérfanas: {len(result.orphan_citations)}")
    print(f"  Cobertura: {result.citation_coverage:.0%}")
    print(f"  Consistencia: {result.style_consistency:.0%}")
    for c in result.inline_citations[:3]:
        print(f"    → {c.style.value}: {c.raw_text} (autor={c.author}, año={c.year})")

    print()
    print("=" * 60)
    print("TEST: Texto IEEE")
    print("=" * 60)
    result2 = detector.analyze(TEXT_IEEE_COMPLETO)
    print(f"  Estilo dominante: {result2.dominant_style.value}")
    print(f"  Citas inline: {len(result2.inline_citations)}")
    print(f"  Entradas bibliográficas: {len(result2.bibliography)}")

    print()
    print("=" * 60)
    print("TEST: Texto sin citas (plagio potencial)")
    print("=" * 60)
    result3 = detector.analyze(TEXT_PLAGIO_SIN_CITAS)
    print(f"  Citas inline: {len(result3.inline_citations)}")
    print(f"  Zonas con cita: {sum(1 for z in result3.zones if z.has_valid_citation)}")
    print(f"  Zonas sin cita: {sum(1 for z in result3.zones if not z.has_valid_citation)}")
